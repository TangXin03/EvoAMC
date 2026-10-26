# encoding=utf-8
import os


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _parse_int_list(value, default):
    if not value:
        return list(default)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _env_int(name, default):
    return int(os.environ.get(name, default))


def get_configs():
    """Return experiment settings with environment-variable overrides.

    The original experiments were tied to one workstation. Keeping the defaults
    here makes the scripts runnable after cloning while still allowing a run to
    select a dataset, SNR, output directory, or GPU pool without editing code.
    """
    data_name = os.environ.get("AMC_EF_DATA_NAME", "rml2016_10a")
    default_classes = {"rml2016_10a": 24, "rml2016_10b": 10, "rml2018_01a": 24}
    classes = _env_int("AMC_EF_CLASSES", default_classes.get(data_name, 24))

    return {
        "data_root": os.environ.get("AMC_EF_DATA_ROOT", os.path.join(PROJECT_ROOT, "data")),
        "result_root": os.environ.get("AMC_EF_RESULT_ROOT", os.path.join(PROJECT_ROOT, "results")),
        "data_name": data_name,
        "classes": classes,
        "fusion_ways": ["add", "mul", "cat", "max", "avg"],
        "fused_nb_feats": _env_int("AMC_EF_FUSED_NB_FEATS", 128),
        "nb_view": _env_int("AMC_EF_NB_VIEW", 5),
        "pop_size": _env_int("AMC_EF_POP_SIZE", 32),
        "nb_iters": _env_int("AMC_EF_NB_ITERS", 10),
        "snr_name": os.environ.get("AMC_EF_SNR_NAME", "view_snr-4"),
        "ablation_name": os.environ.get("AMC_EF_ABLATION", "multi_offspring_predictor"),
        "result_save_dir": os.environ.get("AMC_EF_RESULT_NAME", "RMCL-EA-CAL128-result"),
        "gpu_list": _parse_int_list(os.environ.get("AMC_EF_GPU_LIST"), [0]),
        "epochs": _env_int("AMC_EF_EPOCHS", 15),
        "batch_size": _env_int("AMC_EF_BATCH_SIZE", 256),
        "patience": _env_int("AMC_EF_PATIENCE", 10),
        "is_remove": False,
        "crossover_rate": 0.9,
        "mutation_rate": 0.1,
        "noisy": True,
        "max_len": 40,
    }
