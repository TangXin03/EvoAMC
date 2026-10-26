import copy
import csv
import importlib
import logging
import os
import random
import sys
import time
import types
from warnings import simplefilter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

import convergence_compare_config as compare_config


simplefilter(action="ignore", category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

BASE_CONFIG = compare_config.get_configs()
LOGGER = logging.getLogger("amc_ef_convergence")


def install_compare_config(method_settings=None):
    config_values = dict(BASE_CONFIG)
    if method_settings is not None:
        config_values["ablation_name"] = method_settings["ablation_name"]

    module = types.ModuleType("data_utils.config")
    module.get_configs = lambda: dict(config_values)
    sys.modules["data_utils.config"] = module
    if "data_utils" in sys.modules:
        setattr(sys.modules["data_utils"], "config", module)
    return config_values


install_compare_config()

import gen_offspring_tree
import population_init
import tree_to_strlist
import utils
from code2net_tree import FusionModel
from data_utils.data_uitl import get_views
from gcn_transformer import GCNTransformerPerformancePredictor
from gcnpp import resolve_device, trees_to_graph
from utils_tree import list_to_tree


def arch_str(individual):
    return "+".join([str(item) for item in individual])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(compare_config.DETERMINISTIC)
    torch.backends.cudnn.benchmark = not bool(compare_config.DETERMINISTIC)


def result_dir_for(method_name, seed):
    return os.path.join(
        SCRIPT_DIR,
        compare_config.RESULT_ROOT,
        BASE_CONFIG["data_name"],
        BASE_CONFIG["snr_name"],
        f"seed_{seed}",
        method_name,
    )


def seed_dir_for(seed):
    return os.path.join(
        SCRIPT_DIR,
        compare_config.RESULT_ROOT,
        BASE_CONFIG["data_name"],
        BASE_CONFIG["snr_name"],
        f"seed_{seed}",
    )


def data_dir():
    return os.path.join(compare_config.DATA_ROOT, BASE_CONFIG["data_name"], BASE_CONFIG["snr_name"])


def prepare_result_dir(path):
    os.makedirs(path, exist_ok=True)
    for name in os.listdir(path):
        file_path = os.path.join(path, name)
        if os.path.isfile(file_path):
            os.remove(file_path)


def setup_logging(path):
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    LOGGER.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)

    file_handler = logging.FileHandler(os.path.join(path, "train.log"), encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)


def write_csv_header(path, header):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def append_csv_rows(path, rows):
    if not rows:
        return
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def load_result_acc(result_path):
    acc = {}
    if not os.path.exists(result_path):
        return acc
    with open(result_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            if parts[0] not in acc:
                acc[parts[0]] = float(parts[1])
    return acc


def get_configured_gpu_ids():
    gpu_ids = [int(gpu_id) for gpu_id in BASE_CONFIG.get("gpu_list", [0])]
    if not gpu_ids:
        return [0]
    if not torch.cuda.is_available():
        return [0]

    device_count = torch.cuda.device_count()
    visible = [gpu_id for gpu_id in gpu_ids if gpu_id < device_count]
    if visible:
        return visible

    LOGGER.warning(
        "configured gpu_list=%s, but only %d CUDA device(s) are visible; falling back to cuda:0",
        gpu_ids,
        device_count,
    )
    return [0]


def pick_gpu_for_candidate(candidate_index, generation):
    gpu_ids = get_configured_gpu_ids()
    return gpu_ids[(max(generation, 0) + candidate_index) % len(gpu_ids)]


def get_knowledge_device():
    gpu_ids = get_configured_gpu_ids()
    return resolve_device(f"cuda:{gpu_ids[0]}")


def format_duration(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class MethodRunner:
    def __init__(self, method_name, method_settings, seed, initial_population):
        self.method_name = method_name
        self.method_settings = method_settings
        self.seed = seed
        self.population = copy.deepcopy(initial_population)
        self.result_dir = result_dir_for(method_name, seed)
        self.result_path = os.path.join(self.result_dir, "result_KL.csv")
        self.history_path = os.path.join(self.result_dir, "history_n.csv")
        self.summary_path = os.path.join(self.result_dir, "convergence_summary.csv")
        self.trace_path = os.path.join(self.result_dir, "population_trace.csv")
        self.knowledge_path = os.path.join(self.result_dir, "knowledge_model_performance.csv")
        self.screening_path = os.path.join(self.result_dir, "offspring_screening.csv")
        self.shared_code_sets = set()
        self.global_best = {
            "accuracy": float("-inf"),
            "architecture": None,
            "total_params": 0,
            "best_epoch": -1,
            "model_path": None,
        }
        self.knowledge_model = None

    def prepare(self):
        prepare_result_dir(self.result_dir)
        setup_logging(self.result_dir)
        write_csv_header(
            self.summary_path,
            [
                "method",
                "seed",
                "generation",
                "evaluated_count_total",
                "evaluated_count_generation",
                "population_best_fitness",
                "population_mean_fitness",
                "generation_best_fitness",
                "generation_best_arch",
                "global_best_fitness",
                "global_best_arch",
                "elapsed_seconds",
            ],
        )
        write_csv_header(
            self.trace_path,
            [
                "method",
                "seed",
                "generation",
                "arch",
                "true_fitness",
                "is_population_member",
                "is_newly_evaluated",
                "is_global_best",
            ],
        )
        write_csv_header(
            self.knowledge_path,
            ["method", "seed", "generation", "arch", "estimated_fitness", "true_fitness", "absolute_error", "phase"],
        )
        write_csv_header(
            self.screening_path,
            ["method", "seed", "generation", "pair_index", "rank", "arch", "estimated_fitness", "selected_for_full_eval"],
        )

    def result_file_path(self, filename):
        return os.path.join(self.result_dir, filename)

    def save_global_best_model(self, state_dict, accuracy, architecture, total_params, best_epoch):
        if accuracy <= self.global_best.get("accuracy", float("-inf")):
            return

        model_path = self.result_file_path("best_model.pth")
        meta_path = self.result_file_path("best_model.txt")
        torch.save(
            {
                "architecture": architecture,
                "accuracy": accuracy,
                "total_params": total_params,
                "best_epoch": best_epoch,
                "state_dict": state_dict,
            },
            model_path,
        )

        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(f"architecture: {architecture}\n")
            f.write(f"accuracy: {accuracy:.6f}\n")
            f.write(f"total_params: {total_params}\n")
            f.write(f"best_epoch: {best_epoch}\n")
            f.write(f"model_path: {model_path}\n")

        self.global_best = {
            "accuracy": accuracy,
            "architecture": architecture,
            "total_params": total_params,
            "best_epoch": best_epoch,
            "model_path": model_path,
        }
        LOGGER.info(
            "new global best | acc=%.6f | epoch=%d | params=%d | arch=%s",
            accuracy,
            best_epoch + 1,
            total_params,
            architecture,
        )

    def train_individual(self, individual_code, view_train_x, train_y, view_test_x, test_y, gpu_id):
        device = resolve_device(f"cuda:{gpu_id}", strict_cuda=True)
        individual_code_tree, nb_view = tree_to_strlist.viewfusion(individual_code)
        individual_code_str = arch_str(individual_code)
        LOGGER.info("individual start | device=%s | views=%d | arch=%s", device, nb_view, individual_code_str)

        view_train_xx, view_test_xx = [], []
        for view_idx in individual_code_tree[:nb_view]:
            view_train_xx.append(torch.from_numpy(view_train_x[view_idx]))
            view_test_xx.append(torch.from_numpy(view_test_x[view_idx]))

        _, data_loader_train = utils.create_data_loader(BASE_CONFIG["batch_size"], True, view_train_xx, train_y)
        _, data_loader_test = utils.create_data_loader(BASE_CONFIG["batch_size"], False, view_test_xx, test_y)
        nb_feats = [tensor.shape[1] for tensor in view_train_xx]

        model = FusionModel(individual_code=individual_code_tree, nb_feats=nb_feats, listtree=individual_code)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        best_acc = float("-inf")
        best_epoch = -1
        best_state_dict = None

        curve_path = self.result_file_path(individual_code_str + ".csv")
        with open(curve_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

        for epoch in range(BASE_CONFIG["epochs"]):
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
            best_epoch = BASE_CONFIG["epochs"] - 1
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
        self.save_global_best_model(best_state_dict, final_acc, individual_code_str, total_params, best_epoch)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        LOGGER.info(
            "individual done | acc=%.6f | best_epoch=%d | params=%d",
            final_acc,
            best_epoch + 1,
            total_params,
        )
        return individual_code_str + "," + str(final_acc) + "," + str(total_params)

    def train_candidates(self, generation, candidates, view_train_x, train_y, view_test_x, test_y):
        LOGGER.info(
            "train batch start | generation=%d | candidates=%d | gpu_cycle=%s",
            generation,
            len(candidates),
            get_configured_gpu_ids(),
        )

        result_lines = []
        for idx, candidate in enumerate(candidates):
            candidate_arch = arch_str(candidate)
            LOGGER.info("candidate %d/%d | arch=%s", idx + 1, len(candidates), candidate_arch)
            if candidate_arch in self.shared_code_sets:
                LOGGER.info("candidate status | skip duplicate")
                continue

            gpu_id = pick_gpu_for_candidate(idx, generation)
            self.shared_code_sets.add(candidate_arch)
            LOGGER.info("candidate status | new | gpu=%s", gpu_id)
            result_line = self.train_individual(candidate, view_train_x, train_y, view_test_x, test_y, str(gpu_id))
            utils.write_result_file(result_line, fn=self.result_path)
            result_lines.append(result_line)

        return result_lines

    def split_evaluated(self, candidates):
        acc_map = load_result_acc(self.result_path)
        evaluated, missing = [], []
        for candidate in candidates:
            key = arch_str(candidate)
            if key in acc_map:
                evaluated.append(candidate)
            else:
                missing.append(key)
        return evaluated, missing

    def init_knowledge_model(self):
        if not self.method_settings["use_knowledge_library"]:
            return
        device = get_knowledge_device()
        LOGGER.info("knowledge library model | device=%s | type=GCN+Transformer", device)
        self.knowledge_model = GCNTransformerPerformancePredictor(device=device)

    def build_knowledge_input(self, individuals):
        return trees_to_graph([list_to_tree(individual) for individual in individuals])

    def train_initial_knowledge_model(self):
        if self.knowledge_model is None:
            return
        acc_map = load_result_acc(self.result_path)
        knowledge_input = self.build_knowledge_input(self.population)
        y_train = [acc_map[arch_str(individual)] for individual in self.population]
        LOGGER.info(
            "knowledge train | samples=%d | acc_min=%.6f | acc_max=%.6f | acc_mean=%.6f",
            len(y_train),
            min(y_train),
            max(y_train),
            float(np.mean(y_train)),
        )
        self.knowledge_model.train_predictor(
            knowledge_input,
            y_train,
            epochs=compare_config.KNOWLEDGE_INITIAL_EPOCHS,
        )
        self.record_knowledge_predictions(0, self.population, "initial_population")

    def record_knowledge_predictions(self, generation, individuals, phase):
        if self.knowledge_model is None or not individuals:
            return
        acc_map = load_result_acc(self.result_path)
        knowledge_input = self.build_knowledge_input(individuals)
        self.knowledge_model.eval()
        with torch.no_grad():
            estimated = self.knowledge_model(knowledge_input).detach().cpu().numpy()

        rows = []
        for idx, individual in enumerate(individuals):
            key = arch_str(individual)
            if key not in acc_map:
                continue
            true_fitness = acc_map[key]
            estimate = float(estimated[idx])
            rows.append(
                [
                    self.method_name,
                    self.seed,
                    generation,
                    key,
                    estimate,
                    true_fitness,
                    abs(estimate - true_fitness),
                    phase,
                ]
            )
        append_csv_rows(self.knowledge_path, rows)

    def update_knowledge_model(self, generation, evaluated_offspring):
        if self.knowledge_model is None or not evaluated_offspring:
            return
        self.record_knowledge_predictions(generation, evaluated_offspring, "before_update")
        acc_map = load_result_acc(self.result_path)
        knowledge_input = self.build_knowledge_input(evaluated_offspring)
        y_new = [acc_map[arch_str(individual)] for individual in evaluated_offspring]
        self.knowledge_model.update_predictor(
            knowledge_input,
            y_new,
            epochs=compare_config.KNOWLEDGE_UPDATE_EPOCHS,
        )

    def select_offspring_for_full_eval(self, generation, pair_idx, candidate_offspring):
        k = compare_config.SELECTED_OFFSPRING_PER_PAIR
        if len(candidate_offspring) <= k:
            return candidate_offspring

        if self.knowledge_model is None:
            selected_indices = sorted(random.sample(range(len(candidate_offspring)), k))
            return [candidate_offspring[idx] for idx in selected_indices]

        knowledge_input = self.build_knowledge_input(candidate_offspring)
        self.knowledge_model.eval()
        with torch.no_grad():
            estimated = self.knowledge_model(knowledge_input).detach().cpu().numpy()

        ranked_indices = sorted(range(len(candidate_offspring)), key=lambda idx: float(estimated[idx]), reverse=True)
        selected_indices = set(ranked_indices[:k])
        rows = []
        for rank, idx in enumerate(ranked_indices, start=1):
            rows.append(
                [
                    self.method_name,
                    self.seed,
                    generation,
                    pair_idx,
                    rank,
                    arch_str(candidate_offspring[idx]),
                    float(estimated[idx]),
                    int(idx in selected_indices),
                ]
            )
        append_csv_rows(self.screening_path, rows)
        return [candidate_offspring[idx] for idx in ranked_indices[:k]]

    def record_generation(self, generation, newly_evaluated_arches, elapsed_seconds):
        acc_map = load_result_acc(self.result_path)
        if not acc_map:
            return

        population_keys = [arch_str(individual) for individual in self.population if arch_str(individual) in acc_map]
        population_values = [acc_map[key] for key in population_keys]
        newly_evaluated_arches = [key for key in newly_evaluated_arches if key in acc_map]

        if newly_evaluated_arches:
            generation_best_arch = max(newly_evaluated_arches, key=lambda key: acc_map[key])
            generation_best_fitness = acc_map[generation_best_arch]
        else:
            generation_best_arch = ""
            generation_best_fitness = ""

        global_best_arch = max(acc_map, key=acc_map.get)
        global_best_fitness = acc_map[global_best_arch]
        population_best_fitness = max(population_values) if population_values else ""
        population_mean_fitness = float(np.mean(population_values)) if population_values else ""

        append_csv_rows(
            self.summary_path,
            [
                [
                    self.method_name,
                    self.seed,
                    generation,
                    len(acc_map),
                    len(newly_evaluated_arches),
                    population_best_fitness,
                    population_mean_fitness,
                    generation_best_fitness,
                    generation_best_arch,
                    global_best_fitness,
                    global_best_arch,
                    elapsed_seconds,
                ]
            ],
        )

        trace_keys = sorted(set(population_keys) | set(newly_evaluated_arches))
        rows = []
        for key in trace_keys:
            rows.append(
                [
                    self.method_name,
                    self.seed,
                    generation,
                    key,
                    acc_map[key],
                    int(key in population_keys),
                    int(key in newly_evaluated_arches),
                    int(key == global_best_arch),
                ]
            )
        append_csv_rows(self.trace_path, rows)

    def write_history(self, generation):
        rows = [[generation, arch_str(individual)] for individual in self.population]
        append_csv_rows(self.history_path, rows)

    def write_training_summary(self, start_time, total_seconds):
        end_time = start_time + total_seconds
        with open(self.result_file_path("training_summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"method: {self.method_name}\n")
            f.write(f"seed: {self.seed}\n")
            f.write(f"data_name: {BASE_CONFIG['data_name']}\n")
            f.write(f"snr_name: {BASE_CONFIG['snr_name']}\n")
            f.write(f"pop_size: {BASE_CONFIG['pop_size']}\n")
            f.write(f"nb_iters: {BASE_CONFIG['nb_iters']}\n")
            f.write(f"epochs: {BASE_CONFIG['epochs']}\n")
            f.write(f"batch_size: {BASE_CONFIG['batch_size']}\n")
            f.write(f"use_multi_offspring: {self.method_settings['use_multi_offspring']}\n")
            f.write(f"use_knowledge_library: {self.method_settings['use_knowledge_library']}\n")
            f.write(f"offspring_rounds: {self.method_settings['offspring_rounds']}\n")
            f.write(f"selected_offspring_per_pair: {compare_config.SELECTED_OFFSPRING_PER_PAIR}\n")
            f.write(f"start_time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
            f.write(f"end_time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
            f.write(f"total_seconds: {total_seconds:.2f}\n")
            f.write(f"total_duration: {format_duration(total_seconds)}\n")
            if self.global_best.get("architecture") is not None:
                f.write(f"best_architecture: {self.global_best['architecture']}\n")
                f.write(f"best_accuracy: {self.global_best['accuracy']:.6f}\n")
                f.write(f"best_epoch: {self.global_best['best_epoch']}\n")
                f.write(f"best_model_path: {self.global_best['model_path']}\n")

    def run(self, view_train_x, train_y, view_test_x, test_y):
        self.prepare()
        install_compare_config(self.method_settings)
        LOGGER.info("=" * 72)
        LOGGER.info("convergence comparison start | method=%s | seed=%d", self.method_name, self.seed)
        LOGGER.info("=" * 72)
        LOGGER.info(
            "config | data=%s | snr=%s | pop_size=%d | iters=%d | epochs=%d | batch_size=%d",
            BASE_CONFIG["data_name"],
            BASE_CONFIG["snr_name"],
            BASE_CONFIG["pop_size"],
            BASE_CONFIG["nb_iters"],
            BASE_CONFIG["epochs"],
            BASE_CONFIG["batch_size"],
        )
        LOGGER.info(
            "method | multi_offspring=%s | knowledge_library=%s | offspring_rounds=%d",
            self.method_settings["use_multi_offspring"],
            self.method_settings["use_knowledge_library"],
            self.method_settings["offspring_rounds"],
        )
        LOGGER.info("paths | data=%s | result_dir=%s", data_dir(), self.result_dir)

        start_time = time.time()
        self.init_knowledge_model()

        initial_result_lines = self.train_candidates(
            0,
            self.population,
            view_train_x,
            train_y,
            view_test_x,
            test_y,
        )
        self.write_history(0)
        self.record_generation(
            0,
            [line.split(",")[0] for line in initial_result_lines],
            time.time() - start_time,
        )
        self.train_initial_knowledge_model()

        for generation in tqdm(range(1, BASE_CONFIG["nb_iters"] + 1), desc=self.method_name):
            iter_start = time.time()
            acc_map = load_result_acc(self.result_path)
            self.population, missing_parents = self.split_evaluated(self.population)
            if missing_parents:
                LOGGER.warning(
                    "generation %d | missing parent fitness | count=%d | arches=%s",
                    generation,
                    len(missing_parents),
                    missing_parents,
                )
            if len(self.population) < 2:
                raise RuntimeError(f"Not enough evaluated parents for generation {generation}: {len(self.population)}")

            selected_offspring = []
            pair_count = len(self.population) // 2
            for pair_idx in range(pair_count):
                _, candidate_offspring = gen_offspring_tree.gen_offspring_new(
                    self.population,
                    generation - 1,
                    num_offspring=self.method_settings["offspring_rounds"],
                    result_fn=self.result_path,
                )
                chosen = self.select_offspring_for_full_eval(generation, pair_idx + 1, candidate_offspring)
                selected_offspring.extend(chosen)
                LOGGER.info(
                    "pair %d/%d | generated=%d | selected=%d",
                    pair_idx + 1,
                    pair_count,
                    len(candidate_offspring),
                    len(chosen),
                )

            result_lines = self.train_candidates(
                generation,
                selected_offspring,
                view_train_x,
                train_y,
                view_test_x,
                test_y,
            )
            acc_map = load_result_acc(self.result_path)
            evaluated_offspring, missing_arches = self.split_evaluated(selected_offspring)
            if missing_arches:
                LOGGER.warning(
                    "generation %d | missing evaluated offspring | count=%d | arches=%s",
                    generation,
                    len(missing_arches),
                    missing_arches,
                )

            self.update_knowledge_model(generation, evaluated_offspring)
            self.population = gen_offspring_tree.selection(
                self.population,
                evaluated_offspring,
                self.result_path,
            )
            self.write_history(generation)
            self.record_generation(
                generation,
                [line.split(",")[0] for line in result_lines],
                time.time() - start_time,
            )
            LOGGER.info(
                "generation done | %d/%d | next_pop=%d | total_evaluated=%d | duration=%s",
                generation,
                BASE_CONFIG["nb_iters"],
                len(self.population),
                len(acc_map),
                format_duration(time.time() - iter_start),
            )

        total_seconds = time.time() - start_time
        self.write_training_summary(start_time, total_seconds)
        LOGGER.info("method done | method=%s | duration=%s", self.method_name, format_duration(total_seconds))


def write_initial_population(seed, population):
    os.makedirs(seed_dir_for(seed), exist_ok=True)
    path = os.path.join(seed_dir_for(seed), "initial_population.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "index", "arch"])
        for idx, individual in enumerate(population):
            writer.writerow([seed, idx, arch_str(individual)])
    return path


def merge_method_csv(seed, filename, output_name):
    seed_dir = seed_dir_for(seed)
    output_path = os.path.join(seed_dir, output_name)
    header_written = False
    with open(output_path, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f)
        for method_name in compare_config.METHODS:
            input_path = os.path.join(seed_dir, method_name, filename)
            if not os.path.exists(input_path):
                continue
            with open(input_path, "r", encoding="utf-8") as in_f:
                reader = csv.reader(in_f)
                header = next(reader, None)
                if header and not header_written:
                    writer.writerow(header)
                    header_written = True
                for row in reader:
                    writer.writerow(row)
    return output_path


def merge_seed_csv(filename, output_name):
    snr_dir = os.path.join(
        SCRIPT_DIR,
        compare_config.RESULT_ROOT,
        BASE_CONFIG["data_name"],
        BASE_CONFIG["snr_name"],
    )
    output_path = os.path.join(snr_dir, output_name)
    header_written = False
    with open(output_path, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f)
        for seed in compare_config.SEEDS:
            input_path = os.path.join(snr_dir, f"seed_{seed}", filename)
            if not os.path.exists(input_path):
                continue
            with open(input_path, "r", encoding="utf-8") as in_f:
                reader = csv.reader(in_f)
                header = next(reader, None)
                if header and not header_written:
                    writer.writerow(header)
                    header_written = True
                for row in reader:
                    writer.writerow(row)
    return output_path


def plot_best_individual_curve(summary_csv_path):
    if plt is None:
        LOGGER.warning("matplotlib is not installed; skip best-individual plot generation")
        return None
    if not os.path.exists(summary_csv_path):
        LOGGER.warning("summary csv not found; skip best-individual plot generation | path=%s", summary_csv_path)
        return None

    import pandas as pd

    df = pd.read_csv(summary_csv_path)
    required = {"method", "generation", "population_best_fitness"}
    missing = required - set(df.columns)
    if missing:
        LOGGER.warning("summary csv missing required columns; skip plotting | missing=%s", sorted(missing))
        return None

    df = df.dropna(subset=["generation", "population_best_fitness"])
    if df.empty:
        LOGGER.warning("summary csv has no valid rows for plotting | path=%s", summary_csv_path)
        return None

    df["generation"] = df["generation"].astype(int)
    method_order = ["ours_multi_offspring_knowledge", "baseline_single_offspring"]
    method_labels = {
        "ours_multi_offspring_knowledge": "AMC-EvoMVF",
        "baseline_single_offspring": "Baseline",
    }
    method_styles = {
        "ours_multi_offspring_knowledge": {"color": "#D55E00", "marker": "o", "linestyle": "-"},
        "baseline_single_offspring": {"color": "#0072B2", "marker": "s", "linestyle": "--"},
    }

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "legend.fontsize": 9.3,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "grid.linestyle": "-",
            "lines.linewidth": 2.0,
            "lines.markersize": 4.8,
        }
    )

    fig, ax = plt.subplots(figsize=(4.0, 2.9))
    for method in method_order:
        method_df = df[df["method"] == method].sort_values("generation")
        if method_df.empty:
            continue
        style = method_styles[method]
        markevery = max(1, len(method_df) // 12)
        y_values = method_df["population_best_fitness"] * 100.0
        ax.plot(
            method_df["generation"],
            y_values,
            label=method_labels.get(method, method),
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markevery=markevery,
            markerfacecolor="white",
            markeredgewidth=1.3,
        )

        final_row = method_df.iloc[-1]
        final_y = float(final_row["population_best_fitness"]) * 100.0
        ax.scatter([final_row["generation"]], [final_y], color=style["color"], s=36, zorder=5)
        ax.annotate(
            f"{final_y:.2f}",
            xy=(final_row["generation"], final_y),
            xytext=(-4, 8),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=8.7,
            color=style["color"],
        )

    from matplotlib.ticker import MaxNLocator

    ax.set_title("Best Individual per Generation")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.xaxis.set_major_locator(MaxNLocator(7, integer=True))
    ax.legend(loc="lower right")
    fig.tight_layout()

    output_dir = os.path.dirname(summary_csv_path)
    pdf_path = os.path.join(output_dir, "best_individual_compare.pdf")
    png_path = os.path.join(output_dir, "best_individual_compare.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)

    LOGGER.info("best-individual plot saved | pdf=%s | png=%s", pdf_path, png_path)
    return pdf_path, png_path


def safe_plot_best_individual_curve(summary_csv_path):
    try:
        return plot_best_individual_curve(summary_csv_path)
    except Exception as exc:
        LOGGER.warning(
            "best-individual plot generation failed; training result is kept | path=%s | error=%s",
            summary_csv_path,
            exc,
        )
        return None


def main():
    LOGGER.setLevel(logging.INFO)
    set_seed(compare_config.SEEDS[0])
    view_data_dir = data_dir()
    if not os.path.isdir(view_data_dir):
        raise FileNotFoundError(
            f"View data directory not found: {view_data_dir}. "
            "Set AMC_EF_DATA_ROOT or edit DATA_ROOT in convergence_compare_config.py."
        )

    view_train_x, train_y, view_test_x, test_y = get_views(view_data_dir=view_data_dir)

    for seed in compare_config.SEEDS:
        set_seed(seed)
        initial_population = population_init.generate_population_tree(
            views=len(view_train_x),
            pop_size=BASE_CONFIG["pop_size"],
            verbose=0,
        )
        initial_path = write_initial_population(seed, initial_population)

        for method_name, method_settings in compare_config.METHODS.items():
            set_seed(seed)
            runner = MethodRunner(method_name, method_settings, seed, initial_population)
            LOGGER.info("initial population | seed=%d | path=%s", seed, initial_path)
            runner.run(view_train_x, train_y, view_test_x, test_y)

        merge_method_csv(seed, "convergence_summary.csv", "all_convergence_summary.csv")
        merge_method_csv(seed, "population_trace.csv", "all_population_trace.csv")

    merge_seed_csv("all_convergence_summary.csv", "all_convergence_summary.csv")
    merge_seed_csv("all_population_trace.csv", "all_population_trace.csv")
    safe_plot_best_individual_curve(
        os.path.join(
            SCRIPT_DIR,
            compare_config.RESULT_ROOT,
            BASE_CONFIG["data_name"],
            BASE_CONFIG["snr_name"],
            "all_convergence_summary.csv",
        )
    )


if __name__ == "__main__":
    main()
