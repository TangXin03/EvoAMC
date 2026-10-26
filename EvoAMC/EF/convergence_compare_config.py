import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _parse_int_list(value, default):
    if not value:
        return list(default)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def get_configs():
    return {
        "data_root": os.environ.get("AMC_EF_DATA_ROOT", os.path.join(PROJECT_ROOT, "data")),
        "result_root": os.environ.get(
            "AMC_EF_CC_RESULT_ROOT",
            os.path.join(PROJECT_ROOT, "results", "convergence_compare"),
        ),
        "data_name": os.environ.get("AMC_EF_CC_DATA_NAME", "rml2016_10a"),
        "classes": 24,
        "fusion_ways": ["add", "mul", "cat", "max", "avg"],
        "fused_nb_feats": 128,
        "nb_view": 5,
        "pop_size": 32,
        "nb_iters": int(os.environ.get("AMC_EF_CC_NB_ITERS", "30")),
        "snr_name": os.environ.get("AMC_EF_CC_SNR_NAME", "view_snr-4"),
        "ablation_name": "multi_offspring_predictor",
        "result_save_dir": "convergence_compare",
        "gpu_list": _parse_int_list(os.environ.get("AMC_EF_CC_GPU_LIST"), [0]),
        "epochs": 15,
        "batch_size": 256,
        "patience": 10,
        "is_remove": False,
        "crossover_rate": 0.9,
        "mutation_rate": 0.1,
        "noisy": True,
        "max_len": 40,
    }


DATA_ROOT = os.environ.get("AMC_EF_DATA_ROOT", os.path.join(PROJECT_ROOT, "data"))
RESULT_ROOT = os.environ.get(
    "AMC_EF_CC_RESULT_ROOT",
    os.path.join(PROJECT_ROOT, "results", "convergence_compare"),
)
SEEDS = _parse_int_list(os.environ.get("AMC_EF_CC_SEEDS"), [0])

METHODS = {
    "ours_multi_offspring_knowledge": {
        "ablation_name": "multi_offspring_predictor",
        "use_multi_offspring": True,
        "use_knowledge_library": True,
        "offspring_rounds": 20,
    },
    "baseline_single_offspring": {
        "ablation_name": "single_offspring_baseline",
        "use_multi_offspring": False,
        "use_knowledge_library": False,
        "offspring_rounds": 1,
    },
}

SELECTED_OFFSPRING_PER_PAIR = 2
KNOWLEDGE_INITIAL_EPOCHS = 20
KNOWLEDGE_UPDATE_EPOCHS = 10
DETERMINISTIC = False
