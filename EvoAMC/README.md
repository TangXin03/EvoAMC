# AMC-EvoMVF

This repository contains the implementation of AMC-EvoMVF, a performance-predictor-guided evolutionary search method for multi-view fusion in automatic modulation classification (AMC). The search represents a fusion architecture as a tree, evaluates candidate architectures, and uses a graph predictor to prioritize promising offspring.

## Repository layout

- `EF/`: core evolutionary search, fusion model, graph predictors, and predictor evaluation scripts.
- `EF_transformer/`: Transformer-based predictor entry points, including the multi-GPU evaluator.
- `figures/`: the publication figure script for convergence comparison.
- `data/`: local input data location. It is intentionally not committed.
- `results/`: generated checkpoints, logs, CSV files, and figures. It is intentionally not committed.

The old CIFAR-100 experiment, copied external preprocessing project, papers, review files, cached Python bytecode, model checkpoints, raw arrays, logs, and previous experiment outputs have been removed from the publishable tree.

## Requirements

Use Python 3.10 or newer. Install the packages listed in `requirements.txt`. The PyTorch and PyTorch Geometric wheels must match the CUDA version used by the machine; consult the PyTorch Geometric installation instructions when installing on a GPU server.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## Input data

The search expects precomputed view features. Set `AMC_EF_DATA_ROOT` to a directory containing one subdirectory per dataset and SNR setting:

```text
$AMC_EF_DATA_ROOT/
  rml2016_10a/view_snr-4/
    train_awn.npy  test_awn.npy
    train_mcldnn.npy  test_mcldnn.npy
    train_resnet.npy  test_resnet.npy
    train_lstm.npy  test_lstm.npy
    train_gru.npy  test_gru.npy
    trainy.npy  testy.npy
```

All five view pairs and both label files are required. Each view's first dimension must contain the same number of samples. The raw RML datasets and pretrained backbone models are not included in this repository.

## Run the main search

The defaults target `rml2016_10a` at `view_snr-4`, use GPU 0 when CUDA is available, and write outputs to `results/`. Override settings with environment variables instead of editing source files:

```bash
export AMC_EF_DATA_ROOT=/path/to/precomputed/views
export AMC_EF_DATA_NAME=rml2016_10a
export AMC_EF_SNR_NAME=view_snr-4
export AMC_EF_GPU_LIST=0
python EF/train_EF_GCNPP.py
```

Useful settings include `AMC_EF_ABLATION` (`multi_offspring_predictor`, `multi_offspring_random`, or `single_offspring_baseline`), `AMC_EF_NB_ITERS`, `AMC_EF_EPOCHS`, `AMC_EF_BATCH_SIZE`, and `AMC_EF_RESULT_ROOT`.

The Transformer predictor uses the same data and configuration interface:

```bash
python EF_transformer/train_EF_TransformerPP.py
```

For multi-GPU evaluation, set `AMC_EF_GPU_LIST` to the visible device IDs and run:

```bash
python EF_transformer/train_EF_TransformerPP_parallel.py
```

Convergence comparison scripts are available in `EF/train_convergence_compare.py` and `EF/train_convergence_compare_parallel.py`. They write to `AMC_EF_CC_RESULT_ROOT` (default: `results/convergence_compare`).

## Predictor evaluation

After a search has produced `result_KL.csv`, point the evaluation script at it and run the desired analysis:

```bash
export AMC_EF_RESULT_CSV_PATH=/path/to/result_KL.csv
python EF/evaluate_predictor_holdout.py
```

The other `evaluate_predictor_*.py` scripts provide Transformer, GCN-Transformer, cross-generation, and k-center variants. Evaluation metrics and plots are written next to the selected result file unless an output option in the script is overridden.

## Reproducibility notes

Training is compute-intensive and normally requires a CUDA-enabled PyTorch installation. The repository does not include data, checkpoints, or paper figures so that a clone contains source code only and remains small enough for GitHub.
