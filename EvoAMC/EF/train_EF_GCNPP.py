import csv
import logging
import os
import random
import time
from os.path import join as pjoin
from warnings import simplefilter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import data_utils.config as config
import gen_offspring_tree
import population_init
import tree_to_strlist
import utils
from code2net_tree import FusionModel
from data_utils.data_uitl import get_views
from gcnpp import GCNPerformancePredictor, resolve_device, trees_to_graph
from utils_tree import list_to_tree

logging.getLogger().setLevel(logging.ERROR)
simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

EXPERIMENTS = {
    'multi_offspring_predictor': {
        'use_multi_offspring': True,
        'use_predictor': True,
    },
    'multi_offspring_random': {
        'use_multi_offspring': True,
        'use_predictor': False,
    },
    'single_offspring_baseline': {
        'use_multi_offspring': False,
        'use_predictor': False,
    },
}
MULTI_OFFSPRING_ROUNDS = 20
SINGLE_OFFSPRING_ROUNDS = 1
SELECTED_OFFSPRING_PER_PAIR = 2
RESETTABLE_RESULT_FILES = {
    'best_model.pth',
    'best_model.txt',
    'train.log',
    'training_summary.txt',
}

paras = config.get_configs()
classes = paras['classes']
batch_size = paras['batch_size']
epochs = paras['epochs']
nb_iters = paras['nb_iters']
data_name = paras['data_name']
snr_name = paras['snr_name']

data_base_dir = os.path.join(paras['data_root'], data_name)
view_data_dir = os.path.join(data_base_dir, snr_name)
view_train_x, train_y, view_test_x, test_y = get_views(view_data_dir=view_data_dir)

LOGGER = logging.getLogger("amc_ef")
result_save_dir = ""
GLOBAL_BEST_MODEL = {}


def get_experiment_settings():
    ablation_name = paras.get('ablation_name', 'multi_offspring_predictor')
    if ablation_name not in EXPERIMENTS:
        raise ValueError(
            f"Unsupported ablation_name: {ablation_name}. "
            f"Expected one of: {sorted(EXPERIMENTS)}"
        )

    settings = dict(EXPERIMENTS[ablation_name])
    settings['ablation_name'] = ablation_name
    return settings


def build_result_save_dir():
    experiment = get_experiment_settings()
    return pjoin(
        paras['result_root'],
        data_name + '_view_result',
        'ablation',
        experiment['ablation_name'],
        snr_name,
    )


def result_file_path(filename):
    return os.path.join(result_save_dir, filename)


def prepare_result_dir(path):
    os.makedirs(path, exist_ok=True)
    for name in os.listdir(path):
        file_path = os.path.join(path, name)
        if not os.path.isfile(file_path):
            continue
        if name in RESETTABLE_RESULT_FILES or name.endswith('.csv'):
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

    file_handler = logging.FileHandler(os.path.join(path, 'train.log'), encoding='utf-8')
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)


def log_section(title):
    LOGGER.info("=" * 72)
    LOGGER.info(title)
    LOGGER.info("=" * 72)


def format_arch(architecture, max_len=120):
    text = '+'.join([str(ind) for ind in architecture])
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def get_configured_gpu_ids():
    gpu_list = paras.get('gpu_list', [0])
    if not gpu_list:
        return [0]

    gpu_ids = [int(gpu_id) for gpu_id in gpu_list]
    if not torch.cuda.is_available():
        return [0]

    device_count = torch.cuda.device_count()
    available_gpu_ids = [gpu_id for gpu_id in gpu_ids if gpu_id < device_count]
    if available_gpu_ids:
        return available_gpu_ids

    LOGGER.warning(
        "configured gpu_list=%s, but only %d CUDA device(s) are visible; falling back to cuda:0",
        gpu_ids,
        device_count,
    )
    return [0]


def get_main_device():
    preferred_gpu = get_configured_gpu_ids()[0]
    return resolve_device(f'cuda:{preferred_gpu}')


def pick_gpu_for_candidate(candidate_index, i_iter=0):
    gpu_ids = get_configured_gpu_ids()
    offset = max(i_iter, 0)
    return gpu_ids[(offset + candidate_index) % len(gpu_ids)]


def format_duration(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def reset_global_best_model():
    global GLOBAL_BEST_MODEL
    GLOBAL_BEST_MODEL = {
        'accuracy': float('-inf'),
        'architecture': None,
        'total_params': 0,
        'best_epoch': -1,
        'model_path': None,
    }


def save_global_best_model(state_dict, accuracy, architecture, total_params, best_epoch):
    global GLOBAL_BEST_MODEL

    if accuracy <= GLOBAL_BEST_MODEL.get('accuracy', float('-inf')):
        return

    model_path = result_file_path('best_model.pth')
    meta_path = result_file_path('best_model.txt')
    checkpoint = {
        'architecture': architecture,
        'accuracy': accuracy,
        'total_params': total_params,
        'best_epoch': best_epoch,
        'state_dict': state_dict,
    }
    torch.save(checkpoint, model_path)

    with open(meta_path, 'w', encoding='utf-8') as f:
        f.write(f"architecture: {architecture}\n")
        f.write(f"accuracy: {accuracy:.6f}\n")
        f.write(f"total_params: {total_params}\n")
        f.write(f"best_epoch: {best_epoch}\n")
        f.write(f"model_path: {model_path}\n")

    GLOBAL_BEST_MODEL = {
        'accuracy': accuracy,
        'architecture': architecture,
        'total_params': total_params,
        'best_epoch': best_epoch,
        'model_path': model_path,
    }
    LOGGER.info(
        "new global best | acc=%.6f | epoch=%d | params=%d | arch=%s",
        accuracy,
        best_epoch + 1,
        total_params,
        architecture,
    )


def write_training_summary(start_time, total_seconds, experiment):
    summary_path = result_file_path('training_summary.txt')
    end_time = start_time + total_seconds

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"ablation_name: {experiment['ablation_name']}\n")
        f.write(f"use_multi_offspring: {experiment['use_multi_offspring']}\n")
        f.write(f"use_predictor: {experiment['use_predictor']}\n")
        f.write(f"snr_name: {snr_name}\n")
        f.write(f"start_time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
        f.write(f"end_time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
        f.write(f"total_seconds: {total_seconds:.2f}\n")
        f.write(f"total_duration: {format_duration(total_seconds)}\n")
        if GLOBAL_BEST_MODEL.get('architecture') is not None:
            f.write(f"best_architecture: {GLOBAL_BEST_MODEL['architecture']}\n")
            f.write(f"best_accuracy: {GLOBAL_BEST_MODEL['accuracy']:.6f}\n")
            f.write(f"best_epoch: {GLOBAL_BEST_MODEL['best_epoch']}\n")
            f.write(f"best_model_path: {GLOBAL_BEST_MODEL['model_path']}\n")


def train_individual_pytorch(individual_code, gpu='0'):
    device = resolve_device(f'cuda:{gpu}', strict_cuda=True)
    individual_code_tree, nb_view = tree_to_strlist.viewfusion(individual_code)
    individual_code_str = '+'.join([str(ind) for ind in individual_code])
    LOGGER.info(
        "individual start | device=%s | views=%d | arch=%s",
        device,
        nb_view,
        format_arch(individual_code),
    )

    view_train_xx, view_test_xx = [], []
    for view_idx in individual_code_tree[:nb_view]:
        view_train_xx.append(torch.from_numpy(view_train_x[view_idx]))
        view_test_xx.append(torch.from_numpy(view_test_x[view_idx]))

    dataset_train, data_loader_train = utils.create_data_loader(batch_size, True, view_train_xx, train_y)
    dataset_test, data_loader_test = utils.create_data_loader(batch_size, False, view_test_xx, test_y)
    nb_feats = [tensor.shape[1] for tensor in view_train_xx]
    LOGGER.info(
        "individual data | train=%d | test=%d | feat_dims=%s",
        len(dataset_train),
        len(dataset_test),
        nb_feats,
    )

    model = FusionModel(individual_code=individual_code_tree, nb_feats=nb_feats, listtree=individual_code)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    best_acc = float('-inf')
    best_epoch = -1
    best_state_dict = None

    file_path = os.path.join(result_save_dir, individual_code_str + '.csv')
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])

    for epoch in range(epochs):
        train_loss = 0.0
        correct = 0
        total = 0

        model.train()
        for data_train in data_loader_train:
            x = data_train[:-1]
            y = data_train[-1]
            for idx in range(len(x)):
                x[idx] = x[idx].to(device)

            y = y.to(device)
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
                x = data_test[:-1]
                y = data_test[-1]
                for idx in range(len(x)):
                    x[idx] = x[idx].to(device)

                y = y.to(device)
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

        with open(file_path, mode='a', newline='') as file:
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
            batch_x = data_test[:-1]
            batch_y = data_test[-1]
            for idx in range(len(batch_x)):
                batch_x[idx] = batch_x[idx].to(device)

            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y.squeeze()).sum().item()

    final_acc = correct / total
    save_global_best_model(best_state_dict, final_acc, individual_code_str, total_params, best_epoch)
    LOGGER.info(
        "individual done | acc=%.6f | best_epoch=%d | params=%d",
        final_acc,
        best_epoch + 1,
        total_params,
    )

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return individual_code_str + ',' + str(final_acc) + ',' + str(total_params)


def list2str_tree(list1):
    return '+'.join([str(i) for i in list1])


def single_process_train(i_iter, candidates, shared_code_sets):
    LOGGER.info(
        "train batch start | iter=%d | candidates=%d | gpu_cycle=%s",
        i_iter + 1,
        len(candidates),
        get_configured_gpu_ids(),
    )

    results = []
    for ind_i, candidate in enumerate(candidates):
        LOGGER.info(
            "candidate %d/%d | arch=%s",
            ind_i + 1,
            len(candidates),
            format_arch(candidate),
        )
        code_str = list2str_tree(candidate)
        if code_str in shared_code_sets:
            LOGGER.info("candidate status | skip duplicate")
            continue

        gpu_id = pick_gpu_for_candidate(ind_i, i_iter)
        LOGGER.info("candidate status | new | gpu=%s", gpu_id)
        shared_code_sets.add(code_str)
        result = train_individual_pytorch(candidate, str(gpu_id))
        utils.write_result_file(result, fn=result_file_path('result_KL.csv'))
        results.append(result)

    return results


def select_offspring_for_full_eval(candidate_offspring, predictor, experiment):
    if len(candidate_offspring) <= SELECTED_OFFSPRING_PER_PAIR:
        return candidate_offspring

    if not experiment['use_predictor']:
        selected_indices = sorted(random.sample(range(len(candidate_offspring)), SELECTED_OFFSPRING_PER_PAIR))
        return [candidate_offspring[idx] for idx in selected_indices]

    predictor_trees = [list_to_tree(candidate) for candidate in candidate_offspring]
    predicted = predictor(trees_to_graph(predictor_trees)).detach().cpu().numpy()
    best_idx = int(np.argmax(predicted))
    remaining_indices = [idx for idx in range(len(candidate_offspring)) if idx != best_idx]
    second_idx = random.choice(remaining_indices)
    selected_indices = [best_idx, second_idx]
    return [candidate_offspring[idx] for idx in selected_indices]


def split_evaluated_offspring(candidates, shared_code_acc):
    evaluated = []
    missing = []
    for candidate in candidates:
        arch_str = '+'.join([str(i) for i in candidate])
        if arch_str in shared_code_acc:
            evaluated.append(candidate)
        else:
            missing.append(arch_str)
    return evaluated, missing


def train():
    experiment = get_experiment_settings()
    reset_global_best_model()
    shared_code_sets = set()

    log_section("AMC-EF training start")
    LOGGER.info(
        "config | data=%s | snr=%s | ablation=%s | multi_offspring=%s | predictor=%s",
        data_name,
        snr_name,
        experiment['ablation_name'],
        experiment['use_multi_offspring'],
        experiment['use_predictor'],
    )
    LOGGER.info(
        "config | views=%d | classes=%d | pop_size=%d | iters=%d | epochs=%d | batch_size=%d",
        len(view_train_x),
        classes,
        paras['pop_size'],
        nb_iters,
        epochs,
        batch_size,
    )
    LOGGER.info("paths | view_data=%s | result_dir=%s", view_data_dir, result_save_dir)
    LOGGER.info("data | train=%d | test=%d", len(train_y), len(test_y))

    start = time.time()
    population = population_init.generate_population_tree(views=len(view_train_x), pop_size=paras['pop_size'], verbose=0)

    predictor = None
    predictor_csv_file = result_file_path('predictor_performance.csv')
    if experiment['use_predictor']:
        predictor_device = get_main_device()
        LOGGER.info("predictor | device=%s", predictor_device)
        predictor = GCNPerformancePredictor(device=predictor_device)
        with open(predictor_csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Architecture", "Predicted Accuracy", "Actual Accuracy", "Error"])

    single_process_train(i_iter=-1, candidates=population, shared_code_sets=shared_code_sets)
    for individual in population:
        utils.write_result_file(','.join([str(0), list2str_tree(individual)]), fn=result_file_path('history_n.csv'))

    if predictor is not None:
        predictor_trees = [list_to_tree(individual) for individual in population]
        graph_data = trees_to_graph(predictor_trees)
        shared_code_acc = utils.load_result_n(result_file_path('result_KL.csv'))
        y_train = []
        for individual in population:
            arch_str = '+'.join([str(i) for i in individual])
            y_train.append(shared_code_acc[arch_str])

        LOGGER.info(
            "predictor train | samples=%d | acc_min=%.6f | acc_max=%.6f | acc_mean=%.6f",
            len(y_train),
            min(y_train),
            max(y_train),
            float(np.mean(y_train)),
        )
        predictor.train_predictor(graph_data, y_train, epochs=20)

    offspring_rounds = MULTI_OFFSPRING_ROUNDS if experiment['use_multi_offspring'] else SINGLE_OFFSPRING_ROUNDS

    for i in tqdm(range(nb_iters)):
        iter_start = time.time()
        shared_code_acc = utils.load_result_n(result_file_path('result_KL.csv'))
        population, missing_parents = split_evaluated_offspring(population, shared_code_acc)
        if missing_parents:
            LOGGER.warning(
                "iteration %d | missing parent fitness in result_KL.csv | count=%d | arches=%s",
                i + 1,
                len(missing_parents),
                missing_parents,
            )
        if len(population) < 2:
            raise KeyError(
                "Not enough evaluated parents are available for offspring generation. "
                f"evaluated={len(population)}"
            )
        LOGGER.info("iteration start | %d/%d | parents=%d", i + 1, nb_iters, len(population))
        selected_offspring = []

        for pair_idx in range(len(population) // 2):
            _, candidate_offspring = gen_offspring_tree.gen_offspring_new(
                population,
                i,
                num_offspring=offspring_rounds,
                result_fn=result_file_path('result_KL.csv'),
            )
            chosen_offspring = select_offspring_for_full_eval(candidate_offspring, predictor, experiment)
            selected_offspring.extend(chosen_offspring)
            LOGGER.info(
                "pair %d/%d | generated=%d | selected=%d",
                pair_idx + 1,
                len(population) // 2,
                len(candidate_offspring),
                len(chosen_offspring),
            )

        single_process_train(i_iter=i, candidates=selected_offspring, shared_code_sets=shared_code_sets)
        shared_code_acc = utils.load_result_n(result_file_path('result_KL.csv'))
        evaluated_offspring, missing_arches = split_evaluated_offspring(selected_offspring, shared_code_acc)
        if missing_arches:
            LOGGER.warning(
                "iteration %d | missing evaluated offspring in result_KL.csv | count=%d | arches=%s",
                i + 1,
                len(missing_arches),
                missing_arches,
            )

        if predictor is not None and evaluated_offspring:
            predictor_trees = [list_to_tree(individual) for individual in evaluated_offspring]
            graph_data_pred = trees_to_graph(predictor_trees)
            predicted_acc = predictor(graph_data_pred).detach().cpu().numpy()
            y_new = []
            records = []

            for idx, individual in enumerate(evaluated_offspring):
                arch_str = '+'.join([str(i) for i in individual])
                actual_acc = shared_code_acc[arch_str]
                pred_acc = float(predicted_acc[idx])
                error = abs(pred_acc - actual_acc)
                y_new.append(actual_acc)
                records.append([arch_str, pred_acc, actual_acc, error])

            with open(predictor_csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(records)

            LOGGER.info(
                "iteration eval | candidates=%d | acc_min=%.6f | acc_max=%.6f | acc_mean=%.6f",
                len(y_new),
                min(y_new),
                max(y_new),
                float(np.mean(y_new)),
            )
            predictor.update_predictor(graph_data_pred, y_new, epochs=10)

        population = gen_offspring_tree.selection(
            population,
            evaluated_offspring,
            result_file_path('result_KL.csv'),
        )
        for individual in population:
            utils.write_result_file(','.join([str(i + 1), list2str_tree(individual)]), fn=result_file_path('history_n.csv'))

        LOGGER.info(
            "iteration done | %d/%d | next_pop=%d | duration=%s",
            i + 1,
            nb_iters,
            len(population),
            format_duration(time.time() - iter_start),
        )

    total_seconds = time.time() - start
    write_training_summary(start, total_seconds, experiment)
    LOGGER.info(
        "training done | duration=%s | best_acc=%.6f | best_arch=%s",
        format_duration(total_seconds),
        GLOBAL_BEST_MODEL.get('accuracy', float('-inf')),
        GLOBAL_BEST_MODEL.get('architecture'),
    )


if __name__ == '__main__':
    result_save_dir = build_result_save_dir()
    prepare_result_dir(result_save_dir)
    setup_logging(result_save_dir)
    LOGGER.info("run setup | result_dir=%s | data=%s | snr=%s", result_save_dir, data_name, snr_name)
    train()
