import os

import torch

import train_convergence_compare as base
from convergence_parallel_evaluator import ConvergenceParallelEvaluator


PARALLEL_RESULT_ROOT = os.environ.get(
    "AMC_EF_CONVERGENCE_PARALLEL_RESULT_ROOT",
    os.path.join(base.PROJECT_ROOT, "results", "convergence_compare_parallel"),
)
base.compare_config.RESULT_ROOT = PARALLEL_RESULT_ROOT


class ParallelMethodRunner(base.MethodRunner):
    def __init__(self, method_name, method_settings, seed, initial_population):
        super().__init__(method_name, method_settings, seed, initial_population)
        self.evaluator = None

    def method_config_values(self):
        config_values = dict(base.BASE_CONFIG)
        config_values["ablation_name"] = self.method_settings["ablation_name"]
        return config_values

    def prepare(self):
        super().prepare()
        gpu_ids = base.get_configured_gpu_ids()
        self.evaluator = ConvergenceParallelEvaluator(
            gpu_ids=gpu_ids,
            result_save_dir=self.result_dir,
            config_values=self.method_config_values(),
            deterministic=base.compare_config.DETERMINISTIC,
            seed=self.seed,
        )
        base.LOGGER.info(
            "parallel evaluator | workers=%d | gpu_pool=%s",
            len(gpu_ids),
            gpu_ids,
        )

    def commit_worker_checkpoint(self, result):
        checkpoint_path = result.get("checkpoint_path")
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            return

        try:
            if result["accuracy"] > self.global_best.get("accuracy", float("-inf")):
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                self.save_global_best_model(
                    checkpoint["state_dict"],
                    checkpoint["accuracy"],
                    checkpoint["architecture"],
                    checkpoint["total_params"],
                    checkpoint["best_epoch"],
                )
        finally:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)

    def train_candidates(self, generation, candidates, view_train_x, train_y, view_test_x, test_y):
        if self.evaluator is None:
            raise RuntimeError("Parallel evaluator is not initialized.")

        base.LOGGER.info(
            "parallel train batch start | generation=%d | candidates=%d | gpu_pool=%s",
            generation,
            len(candidates),
            self.evaluator.gpu_ids,
        )

        pending_candidates = []
        for idx, candidate in enumerate(candidates):
            candidate_arch = base.arch_str(candidate)
            base.LOGGER.info("candidate %d/%d | arch=%s", idx + 1, len(candidates), candidate_arch)
            if candidate_arch in self.shared_code_sets:
                base.LOGGER.info("candidate status | skip duplicate")
                continue

            assigned_gpu = self.evaluator.assigned_gpu_for_index(len(pending_candidates))
            self.shared_code_sets.add(candidate_arch)
            pending_candidates.append(
                {
                    "candidate": candidate,
                    "arch_str": candidate_arch,
                    "assigned_gpu": assigned_gpu,
                }
            )
            base.LOGGER.info("candidate status | queued | target_gpu=%s", assigned_gpu)

        if not pending_candidates:
            return []

        results = self.evaluator.evaluate_candidates(pending_candidates)
        result_lines = []
        for result in results:
            candidate_arch = result["arch_str"]
            if result["status"] != "ok":
                self.shared_code_sets.discard(candidate_arch)
                base.LOGGER.error(
                    "candidate failed | arch=%s | gpu=%s | error=%s",
                    candidate_arch,
                    result.get("gpu_id"),
                    result.get("error"),
                )
                if result.get("traceback"):
                    base.LOGGER.error("candidate traceback | arch=%s\n%s", candidate_arch, result["traceback"])
                continue

            base.utils.write_result_file(result["result_line"], fn=self.result_path)
            self.commit_worker_checkpoint(result)
            result_lines.append(result["result_line"])
            base.LOGGER.info(
                "candidate done | arch=%s | gpu=%s | acc=%.6f | best_epoch=%d | params=%d | duration=%s",
                candidate_arch,
                result["gpu_id"],
                result["accuracy"],
                result["best_epoch"] + 1,
                result["total_params"],
                base.format_duration(result["duration_seconds"]),
            )

        return result_lines

    def run(self, view_train_x, train_y, view_test_x, test_y):
        try:
            return super().run(view_train_x, train_y, view_test_x, test_y)
        finally:
            if self.evaluator is not None:
                self.evaluator.close()


def main():
    base.LOGGER.setLevel(base.logging.INFO)
    base.set_seed(base.compare_config.SEEDS[0])
    view_data_dir = base.data_dir()
    if not os.path.isdir(view_data_dir):
        raise FileNotFoundError(
            f"View data directory not found: {view_data_dir}. "
            "Set AMC_EF_DATA_ROOT or edit DATA_ROOT in convergence_compare_config.py."
        )

    view_train_x, train_y, view_test_x, test_y = base.get_views(view_data_dir=view_data_dir)

    for seed in base.compare_config.SEEDS:
        base.set_seed(seed)
        initial_population = base.population_init.generate_population_tree(
            views=len(view_train_x),
            pop_size=base.BASE_CONFIG["pop_size"],
            verbose=0,
        )
        initial_path = base.write_initial_population(seed, initial_population)

        for method_name, method_settings in base.compare_config.METHODS.items():
            base.set_seed(seed)
            runner = ParallelMethodRunner(method_name, method_settings, seed, initial_population)
            base.LOGGER.info("initial population | seed=%d | path=%s", seed, initial_path)
            runner.run(view_train_x, train_y, view_test_x, test_y)

        base.merge_method_csv(seed, "convergence_summary.csv", "all_convergence_summary.csv")
        base.merge_method_csv(seed, "population_trace.csv", "all_population_trace.csv")

    base.merge_seed_csv("all_convergence_summary.csv", "all_convergence_summary.csv")
    base.merge_seed_csv("all_population_trace.csv", "all_population_trace.csv")
    plot_path = os.path.join(
        base.SCRIPT_DIR,
        base.compare_config.RESULT_ROOT,
        base.BASE_CONFIG["data_name"],
        base.BASE_CONFIG["snr_name"],
        "all_convergence_summary.csv",
    )
    plot_fn = getattr(base, "safe_plot_best_individual_curve", None)
    if callable(plot_fn):
        plot_fn(plot_path)
    else:
        base.LOGGER.warning(
            "best-individual plot helper is unavailable in base module; skip plotting | path=%s",
            plot_path,
        )


if __name__ == "__main__":
    main()
