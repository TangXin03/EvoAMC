import numpy as np
import os
import logging
try:
    from . import config
except ImportError:
    import config

opt = os.path
paras = config.get_configs()
nb_view = paras['nb_view']
LOGGER = logging.getLogger("amc_ef")

add_noise = True  # 是否添加噪声
noise_mean = 0  # 噪声均值
noise_std = 5

def add_gaussian_noise(data, mean=0, std=1.0):
    """
    向数据添加高斯噪声
    :param data: 原始数据数组
    :param mean: 噪声均值
    :param std: 噪声标准差
    :return: 添加噪声后的数据
    """
    noise = np.random.normal(mean, std, data.shape)
    noisy_data = data + noise
    return noisy_data

def get_views(view_data_dir='view_new'):
    # models_ls = ['awn', 'amc', 'dual', 'mcldnn', 'cldnn', 'resnet', 'lstm', 'gru']
    # models_ls = ['awn','mcldnn', 'resnet', 'lstm', 'gru']
    models_ls = ['awn','mcldnn', 'resnet', 'lstm', 'gru']

    # models_ls = ["resnet",'lstm', 'gru']

    view_train_x = []  ## 训练视图数据
    view_test_x = []  ## 测试视图数据
    for i, model in enumerate(models_ls):
        train_data = np.load(os.path.join(view_data_dir,'train_' + model + '.npy'))  # 读取npy的数据
        test_data = np.load(os.path.join(view_data_dir, 'test_' + model + '.npy'))

        # 为后五个视图添加噪声
        # if add_noise and i>=3:
            # train_data = add_gaussian_noise(train_data, mean=noise_mean, std=noise_std)
            # test_data = add_gaussian_noise(test_data, mean=noise_mean, std=noise_std)

        view_train_x.append(train_data)
        view_test_x.append(test_data)
    train_y = np.load(os.path.join(view_data_dir, 'trainy.npy'))
    test_y = np.load(os.path.join(view_data_dir, 'testy.npy'))
    LOGGER.info("loaded views | train=%d | test=%d | dir=%s",
                len(train_y), len(test_y), view_data_dir)
    return view_train_x, train_y, view_test_x, test_y  ## 得到视图

def get_views_snr(view_data_dir='view_new'):
    models_ls = ['awn','mcldnn', 'resnet', 'lstm', 'gru']

    # models_ls = ["resnet",'lstm', 'gru']

    view_train_x = []  ## 训练视图数据
    view_test_x = []  ## 测试视图数据
    for i, model in enumerate(models_ls):
        train_data = np.load(os.path.join(view_data_dir,'train_' + model + '.npy'))  # 读取npy的数据
        test_data = np.load(os.path.join(view_data_dir, 'test_' + model + '.npy'))
        view_train_x.append(train_data)
        view_test_x.append(test_data)
    train_y = np.load(os.path.join(view_data_dir, 'trainy.npy'))
    test_y = np.load(os.path.join(view_data_dir, 'testy.npy'))
    train_snr = np.load(os.path.join(view_data_dir, 'train_snr.npy'))
    test_snr = np.load(os.path.join(view_data_dir, 'test_snr.npy'))

    LOGGER.info("loaded views with snr | train=%d | test=%d | dir=%s",
                len(train_y), len(test_y), view_data_dir)
    return view_train_x, train_y,train_snr, view_test_x, test_y,test_snr  ## 得到视图

if __name__ == '__main__':
    pass
