import importlib.util
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SOURCE_EF_DIR = os.path.join(PROJECT_ROOT, "EF")

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if SOURCE_EF_DIR not in sys.path:
    sys.path.insert(1, SOURCE_EF_DIR)


def load_source_training_module():
    source_path = os.path.join(SOURCE_EF_DIR, "train_EF_GCNPP.py")
    spec = importlib.util.spec_from_file_location("ef_transformer_train_module", source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_result_save_dir(module):
    return module.build_result_save_dir()


def main():
    module = load_source_training_module()
    module.result_save_dir = build_result_save_dir(module)
    module.prepare_result_dir(module.result_save_dir)
    module.setup_logging(module.result_save_dir)
    module.LOGGER.info(
        "run setup | result_dir=%s | data=%s | snr=%s | predictor=transformer",
        module.result_save_dir,
        module.data_name,
        module.snr_name,
    )
    module.train()


if __name__ == "__main__":
    main()
