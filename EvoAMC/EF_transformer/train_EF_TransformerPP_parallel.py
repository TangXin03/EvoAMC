import importlib.util
import os
import sys

import torch

from parallel_evaluator import ParallelEvaluator


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SOURCE_EF_DIR = os.path.join(PROJECT_ROOT, "EF")
PARALLEL_CONFIG_PATH = os.path.join(SCRIPT_DIR, "parallel_config.py")

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if SOURCE_EF_DIR not in sys.path:
    sys.path.insert(1, SOURCE_EF_DIR)


def install_parallel_config_module():
    module_name = "data_utils.config"
    spec = importlib.util.spec_from_file_location(module_name, PARALLEL_CONFIG_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    if "data_utils" in sys.modules:
        setattr(sys.modules["data_utils"], "config", module)
    return module


def load_source_training_module():
    install_parallel_config_module()
    source_path = os.path.join(SOURCE_EF_DIR, "train_EF_GCNPP.py")
    spec = importlib.util.spec_from_file_location("ef_transformer_train_parallel_module", source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_result_save_dir(module):
    return module.build_result_save_dir()


def remove_file_if_exists(path):
    if os.path.isfile(path):
        os.remove(path)


def commit_worker_checkpoint(module, result):
    checkpoint_path = result.get("checkpoint_path")
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return

    try:
        if result["accuracy"] > module.GLOBAL_BEST_MODEL.get("accuracy", float("-inf")):
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            module.save_global_best_model(
                checkpoint["state_dict"],
                checkpoint["accuracy"],
                checkpoint["architecture"],
                checkpoint["total_params"],
                checkpoint["best_epoch"],
            )
    finally:
        remove_file_if_exists(checkpoint_path)


def build_parallel_single_process_train(module, evaluator):
    def parallel_single_process_train(i_iter, candidates, shared_code_sets):
        module.LOGGER.info(
            "train batch start | iter=%d | candidates=%d | gpu_pool=%s",
            i_iter + 1,
            len(candidates),
            evaluator.gpu_ids,
        )

        pending_candidates = []
        for ind_i, candidate in enumerate(candidates):
            module.LOGGER.info(
                "candidate %d/%d | arch=%s",
                ind_i + 1,
                len(candidates),
                module.format_arch(candidate),
            )
            code_str = module.list2str_tree(candidate)
            if code_str in shared_code_sets:
                module.LOGGER.info("candidate status | skip duplicate")
                continue

            shared_code_sets.add(code_str)
            pending_candidates.append(
                {
                    "candidate": candidate,
                    "arch_str": code_str,
                    "assigned_gpu": evaluator.assigned_gpu_for_index(len(pending_candidates)),
                }
            )
            module.LOGGER.info(
                "candidate status | queued | target_gpu=%s",
                pending_candidates[-1]["assigned_gpu"],
            )

        if not pending_candidates:
            return []

        results = evaluator.evaluate_candidates(pending_candidates)
        result_lines = []
        for result in results:
            arch_str = result["arch_str"]
            if result["status"] != "ok":
                shared_code_sets.discard(arch_str)
                module.LOGGER.error(
                    "candidate failed | arch=%s | gpu=%s | error=%s",
                    arch_str,
                    result.get("gpu_id"),
                    result.get("error"),
                )
                if result.get("traceback"):
                    module.LOGGER.error("candidate traceback | arch=%s\n%s", arch_str, result["traceback"])
                continue

            module.utils.write_result_file(
                result["result_line"],
                fn=module.result_file_path("result_KL.csv"),
            )
            commit_worker_checkpoint(module, result)
            result_lines.append(result["result_line"])
            module.LOGGER.info(
                "candidate done | arch=%s | gpu=%s | acc=%.6f | best_epoch=%d | params=%d | duration=%s",
                arch_str,
                result["gpu_id"],
                result["accuracy"],
                result["best_epoch"] + 1,
                result["total_params"],
                module.format_duration(result["duration_seconds"]),
            )

        return result_lines

    return parallel_single_process_train


def build_safe_selection(module):
    def safe_selection(P_t, Q_t, result_fn=None):
        shared_code_acc = (
            module.utils.load_result_n(result_fn)
            if result_fn is not None
            else module.utils.load_result_n()
        )

        def arch_str(candidate):
            return "+".join([str(i) for i in candidate])

        filtered_parents = [candidate for candidate in P_t if arch_str(candidate) in shared_code_acc]
        filtered_offspring = [candidate for candidate in Q_t if arch_str(candidate) in shared_code_acc]

        dropped_parents = len(P_t) - len(filtered_parents)
        dropped_offspring = len(Q_t) - len(filtered_offspring)
        if dropped_parents or dropped_offspring:
            module.LOGGER.warning(
                "selection filter | dropped_parents=%d | dropped_offspring=%d",
                dropped_parents,
                dropped_offspring,
            )

        target_size = len(P_t)
        select_pool = filtered_parents + filtered_offspring
        if not select_pool:
            raise KeyError(
                "No evaluated candidates are available for selection. "
                "Both parent and offspring pools are missing from result_KL.csv."
            )

        if len(select_pool) == 1:
            return [select_pool[0] for _ in range(target_size)]

        def select_one():
            two = module.random.sample(range(len(select_pool)), 2)
            a1 = arch_str(select_pool[two[0]])
            a2 = arch_str(select_pool[two[1]])
            return select_pool[two[0]] if shared_code_acc[a1] > shared_code_acc[a2] else select_pool[two[1]]

        next_population = []
        while len(next_population) < target_size:
            next_population.append(select_one())

        max_code_str = max(shared_code_acc, key=shared_code_acc.get)
        max_code = max_code_str.strip().split("+")
        min_code_str = min(shared_code_acc, key=shared_code_acc.get)

        has_global_best = False
        for candidate in next_population:
            if arch_str(candidate) == max_code_str:
                has_global_best = True
                break

        if not has_global_best:
            min_index = 0
            for idx, candidate in enumerate(next_population):
                if arch_str(candidate) == min_code_str:
                    min_index = idx
                    break
            next_population[min_index] = max_code

        return next_population

    return safe_selection


def force_predictor_on_cpu(module):
    def get_cpu_device():
        return module.resolve_device("cpu")

    module.get_main_device = get_cpu_device


def main():
    evaluator = None
    module = load_source_training_module()
    module.result_save_dir = build_result_save_dir(module)
    module.prepare_result_dir(module.result_save_dir)
    module.setup_logging(module.result_save_dir)

    gpu_ids = module.get_configured_gpu_ids()
    evaluator = ParallelEvaluator(gpu_ids=gpu_ids, result_save_dir=module.result_save_dir)
    module.single_process_train = build_parallel_single_process_train(module, evaluator)
    module.gen_offspring_tree.selection = build_safe_selection(module)
    force_predictor_on_cpu(module)

    module.LOGGER.info(
        "run setup | result_dir=%s | data=%s | snr=%s | predictor=transformer | parallel_eval=True | gpu_pool=%s | config=%s",
        module.result_save_dir,
        module.data_name,
        module.snr_name,
        gpu_ids,
        PARALLEL_CONFIG_PATH,
    )

    try:
        module.train()
    finally:
        if evaluator is not None:
            evaluator.close()


if __name__ == "__main__":
    main()
