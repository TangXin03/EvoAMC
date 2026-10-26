# encoding=utf-8
# author: liang xinyan
# email: liangxinyan48@163.com
import torch
import os
import data_utils.config as config
from torch.utils.data import TensorDataset, DataLoader

paras = config.get_configs()
data_name = paras['data_name']
idxx = 0


def _default_result_path(filename):
    return os.path.join(
        paras['result_root'],
        data_name + '_view_result',
        'ablation',
        paras.get('ablation_name', 'multi_offspring_predictor'),
        paras['snr_name'],
        filename,
    )


def get_nb_view_by_individal_code(code):
    nb_view = (len(code) + 1) // 2  # 视图个数
    return nb_view


def write_result_file(str, fn='result.csv'):
    with open(fn, 'a+') as f:
        f.write(str)
        f.write('\n')
        f.flush()


def load_result(result_fn=None):
    result_fn = result_fn or _default_result_path('result.csv')
    shared_code_acc = {}  # 这是一个字典
    shared_code_acc_set = set()  ## 保存每一种融合方式的 精度
    with open(result_fn) as f:
        for item in f.readlines():
            items = item.strip().split(',')
            if items[0] not in shared_code_acc_set:
                shared_code_acc[items[0]] = float(items[1])  ## 返回每一种方式的精度
    return shared_code_acc  ##  得到了所有方式的精度  返回的是一个字典

def load_result_n(result_fn=None):
    result_fn = result_fn or _default_result_path('result_KL.csv')
    shared_code_acc = {}  # 这是一个字典
    shared_code_acc_set = set()  ## 保存每一种融合方式的 精度
    with open(result_fn) as f:
        for item in f.readlines():
            items = item.strip().split(',')
            if items[0] not in shared_code_acc_set:
                shared_code_acc[items[0]] = float(items[1])  ## 返回每一种方式的精度
    return shared_code_acc  ##  得到了所有方式的精度  返回的是一个字典


def load_result_not_select(result_fn=None):
    result_fn = result_fn or _default_result_path('result_not_select.csv')
    shared_code_acc = {}  # 这是一个字典
    shared_code_acc_set = set()  ## 保存每一种融合方式的 精度
    with open(result_fn) as f:
        for item in f.readlines():
            items = item.strip().split(',')
            if items[0] not in shared_code_acc_set:
                shared_code_acc[items[0]] = float(items[1])  ## 返回每一种方式的精度
    return shared_code_acc  ##  得到了所有方式的精度  返回的是一个字典


def load_individual_result(code_str, result_save_dir):
    """
    从 result_KL.csv 中读取个体的精度和不确定性
    """
    result_file = os.path.join(result_save_dir, 'result_KL.csv')

    if not os.path.exists(result_file):
        return None

    with open(result_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            # 新格式：arch_str,acc,unc,ece
            if len(parts) >= 4 and parts[0] == code_str:
                return {
                    'acc': float(parts[1]),
                    'unc': float(parts[2]),
                    'ece': float(parts[3])
                }
            # 兼容旧格式：arch_str,acc
            elif len(parts) == 2 and parts[0] == code_str:
                return {
                    'acc': float(parts[1]),
                    'unc': 0.0,  # 旧数据没有不确定性，默认0
                    'ece': 0.0
                }

    # 没找到，返回None
    return None

"""
Python中str.join()方法用于将序列中的元素以指定的字符连接生成一个新的字符串。其中，序列中的元素应是字符串类型。
"""


def list2str(list1):
    return '-'.join([str(i) for i in list1])


def sign_sqrt(x):
    return torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-10)


def l2_norm(x):
    return torch.nn.functional.normalize(x, p=2, dim=-1)


### 这里写一个数据迭代器 多模态数据迭代器

def create_data_loader(batch_size, shuffle, data, labels=None):
    """

    :param batch_size:  数据一次的多少
    :param shuffle:     是否打乱
    :param data:        数据模态个数 还有它的标签
    :return:            返回数据集合  和   迭代器
    data中包含所有的模态数据，每个数据都是一个张量
    假设每个数据的第一维度都是样本数量，需要保证它们相等
    循环遍历所有张量，将它们存储到列表中
    """
    tensors = []
    for tensor in data:
        tensors.append(tensor)

    # 如果有标签数据，则将标签数据也加入到列表中

    if labels is not None:
        tensors.append(torch.tensor(labels))

    # 这里要使用*操作符，将列表中的所有张量作为参数传入TensorDataset 对象中
    dataset = TensorDataset(*tensors)

    # 创建DataLoader对象
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # 返回对象
    return dataset, data_loader


if __name__ == '__main__':
    x = torch.randn(size=(5, 5))
    print(x)
    x = sign_sqrt(x)
    print(x)
