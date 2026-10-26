import csv
import hashlib
import os
import queue
import random
import sys
import time
import traceback
import types
from dataclasses import dataclass

import numpy as np
import torch
import torch.multiprocessing as mp


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_CHECKPOINT_DIRNAME = "_parallel_tmp"


def configure_import_paths():
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)


def install_config_values(config_values):
    module = types.ModuleType("data_utils.config")
    module.get_configs = lambda: dict(config_values)
    sys.modules["data_utils.config"] = module
    if "data_utils" in sys.modules:
        setattr(sys.modules["data_utils"], "config", module)


@dataclass
class WorkerRuntime:
    config_values: dict
    deterministic: bool
    seed: int
    utils: object
    tree_to_strlist: object
    fusion_model_cls: object
    resolve_device_fn: object
    view_train_x: object
    train_y: object
    view_test_x: object
    test_y: object
    result_save_dir: str


def build_worker_runtime(result_save_dir, config_values, deterministic, seed):
    configure_import_paths()
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    install_config_values(config_values)

    import convergence_compare_config as compare_config
    import tree_to_strlist
    import utils
    from code2net_tree import FusionModel
    from data_utils.data_uitl import get_views
    from gcnpp import resolve_device

    view_data_dir = os.path.join(
        compare_config.DATA_ROOT,
        config_values["data_name"],
        config_values["snr_name"],
    )
    view_train_x, train_y, view_test_x, test_y = get_views(view_data_dir=view_data_dir)

    return WorkerRuntime(
        config_values=config_values,
        deterministic=deterministic,
        seed=seed,
        utils=utils,
        tree_to_strlist=tree_to_strlist,
        fusion_model_cls=FusionModel,
        resolve_device_fn=resolve_device,
        view_train_x=view_train_x,
        train_y=train_y,
        view_test_x=view_test_x,
        test_y=test_y,
        result_save_dir=result_save_dir,
    )


def build_checkpoint_path(result_save_dir, worker_name, job_id, arch_str):
    checkpoint_dir = os.path.join(result_save_dir, TEMP_CHECKPOINT_DIRNAME)
    os.makedirs(checkpoint_dir, exist_ok=True)
    arch_hash = hashlib.md5(arch_str.encode("utf-8")).hexdigest()[:12]
    filename = f"{worker_name}_job{job_id}_{arch_hash}.pth"
    return os.path.join(checkpoint_dir, filename)


def seed_for_arch(base_seed, arch_str):
    arch_hash = int(hashlib.md5(arch_str.encode("utf-8")).hexdigest()[:8], 16)
    return (int(base_seed) * 1000003 + arch_hash) % (2**31 - 1)


def set_task_seed(base_seed, arch_str, deterministic):
    seed = seed_for_arch(base_seed, arch_str)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def train_individual_on_worker(candidate, gpu_id, job_id, runtime):
    import torch.nn as nn
    import torch.optim as optim

    arch_str = "+".join([str(item) for item in candidate])
    set_task_seed(runtime.seed, arch_str, runtime.deterministic)

    epochs = runtime.config_values["epochs"]
    batch_size = runtime.config_values["batch_size"]
    device = runtime.resolve_device_fn(f"cuda:{gpu_id}", strict_cuda=True)
    individual_code_tree, nb_view = runtime.tree_to_strlist.viewfusion(candidate)

    view_train_xx, view_test_xx = [], []
    for view_idx in individual_code_tree[:nb_view]:
        view_train_xx.append(torch.from_numpy(runtime.view_train_x[view_idx]))
        view_test_xx.append(torch.from_numpy(runtime.view_test_x[view_idx]))

    _, data_loader_train = runtime.utils.create_data_loader(
        batch_size,
        True,
        view_train_xx,
        runtime.train_y,
    )
    _, data_loader_test = runtime.utils.create_data_loader(
        batch_size,
        False,
        view_test_xx,
        runtime.test_y,
    )
    nb_feats = [tensor.shape[1] for tensor in view_train_xx]

    model = runtime.fusion_model_cls(
        individual_code=individual_code_tree,
        nb_feats=nb_feats,
        listtree=candidate,
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    best_acc = float("-inf")
    best_epoch = -1
    best_state_dict = None

    curve_path = os.path.join(runtime.result_save_dir, arch_str + ".csv")
    with open(curve_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    start_time = time.time()
    for epoch in range(epochs):
        train_loss = 0.0
        correct = 0
        total = 0

        model.train()
        for data_train in data_loader_train:
            x = list(data_train[:-1])
            y = data_train[-1].to(device)
            for idx in range(len(x)):
                x[idx] = x[idx].to(device)

            optimizer.zero_grad()
            predictions = model(x)
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x[0].size(0)
            total += y.size(0)
            _, predicted = torch.max(predictions.data, 1)
            correct += (predicted == y.squeeze()).sum().item()

        epoch_loss_train = train_loss / float(len(data_loader_train.dataset))
        epoch_acc_train = correct / total

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            correct = 0
            total = 0
            for data_test in data_loader_test:
                x = list(data_test[:-1])
                y = data_test[-1].to(device)
                for idx in range(len(x)):
                    x[idx] = x[idx].to(device)

                predictions = model(x)
                loss = criterion(predictions, y)
                val_loss += loss.item() * x[0].size(0)
                total += y.size(0)
                _, predicted = torch.max(predictions.data, 1)
                correct += (predicted == y.squeeze()).sum().item()

            epoch_loss_test = val_loss / float(len(data_loader_test.dataset))
            epoch_acc_test = correct / total
            if epoch_acc_test > best_acc:
                best_acc = epoch_acc_test
                best_epoch = epoch
                best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

        with open(curve_path, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([epoch, epoch_loss_train, epoch_acc_train, epoch_loss_test, epoch_acc_test])

    if best_state_dict is None:
        best_epoch = epochs - 1
        best_state_dict = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

    model.load_state_dict(best_state_dict)
    total_params = sum(p.numel() for p in model.parameters())
    correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for data_test in data_loader_test:
            batch_x = list(data_test[:-1])
            batch_y = data_test[-1].to(device)
            for idx in range(len(batch_x)):
                batch_x[idx] = batch_x[idx].to(device)

            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y.squeeze()).sum().item()

    final_acc = correct / total
    checkpoint_path = build_checkpoint_path(
        runtime.result_save_dir,
        f"gpu{gpu_id}",
        job_id,
        arch_str,
    )
    checkpoint = {
        "architecture": arch_str,
        "accuracy": final_acc,
        "total_params": total_params,
        "best_epoch": best_epoch,
        "state_dict": best_state_dict,
    }
    torch.save(checkpoint, checkpoint_path)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "status": "ok",
        "arch_str": arch_str,
        "candidate": candidate,
        "accuracy": final_acc,
        "total_params": total_params,
        "best_epoch": best_epoch,
        "gpu_id": gpu_id,
        "checkpoint_path": checkpoint_path,
        "result_line": f"{arch_str},{final_acc},{total_params}",
        "duration_seconds": time.time() - start_time,
    }


def worker_main(gpu_id, task_queue, result_queue, result_save_dir, config_values, deterministic, seed):
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    runtime = build_worker_runtime(result_save_dir, config_values, deterministic, seed)

    while True:
        task = task_queue.get()
        if task is None:
            break

        try:
            result = train_individual_on_worker(
                task["candidate"],
                gpu_id,
                task["job_id"],
                runtime,
            )
        except Exception as exc:
            result = {
                "status": "failed",
                "arch_str": task["arch_str"],
                "candidate": task["candidate"],
                "gpu_id": gpu_id,
                "job_id": task["job_id"],
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        else:
            result["job_id"] = task["job_id"]

        result_queue.put(result)


class ConvergenceParallelEvaluator:
    def __init__(self, gpu_ids, result_save_dir, config_values, deterministic, seed):
        if not gpu_ids:
            raise ValueError("gpu_ids must not be empty.")

        self.gpu_ids = list(gpu_ids)
        self.result_save_dir = result_save_dir
        self.config_values = dict(config_values)
        self.deterministic = bool(deterministic)
        self.seed = int(seed)
        self.ctx = mp.get_context("spawn")
        self.task_queues = []
        self.result_queue = self.ctx.Queue()
        self.workers = []
        self.next_job_id = 0

        checkpoint_dir = os.path.join(self.result_save_dir, TEMP_CHECKPOINT_DIRNAME)
        os.makedirs(checkpoint_dir, exist_ok=True)
        for name in os.listdir(checkpoint_dir):
            file_path = os.path.join(checkpoint_dir, name)
            if os.path.isfile(file_path):
                os.remove(file_path)

        for gpu_id in self.gpu_ids:
            task_queue = self.ctx.Queue()
            process = self.ctx.Process(
                target=worker_main,
                args=(
                    gpu_id,
                    task_queue,
                    self.result_queue,
                    self.result_save_dir,
                    self.config_values,
                    self.deterministic,
                    self.seed,
                ),
                daemon=True,
            )
            process.start()
            self.task_queues.append(task_queue)
            self.workers.append(process)

    def assigned_gpu_for_index(self, candidate_index):
        return self.gpu_ids[candidate_index % len(self.gpu_ids)]

    def evaluate_candidates(self, candidates):
        if not candidates:
            return []

        pending_count = 0
        for candidate_index, candidate in enumerate(candidates):
            gpu_index = candidate_index % len(self.gpu_ids)
            gpu_id = self.gpu_ids[gpu_index]
            task = {
                "job_id": self.next_job_id,
                "candidate": candidate["candidate"],
                "arch_str": candidate["arch_str"],
                "assigned_gpu": gpu_id,
            }
            self.next_job_id += 1
            self.task_queues[gpu_index].put(task)
            pending_count += 1

        results = []
        while len(results) < pending_count:
            try:
                result = self.result_queue.get(timeout=5)
            except queue.Empty:
                dead_workers = [worker.pid for worker in self.workers if not worker.is_alive()]
                if dead_workers:
                    raise RuntimeError(f"Parallel worker died unexpectedly: {dead_workers}")
                continue
            results.append(result)

        results.sort(key=lambda item: item["job_id"])
        return results

    def close(self):
        for task_queue in self.task_queues:
            task_queue.put(None)

        for worker in self.workers:
            worker.join(timeout=10)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
