# encoding=utf-8
# author: liang xinyan
# email: liangxinyan48@163.com
# encoding=utf-8
# author: liang xinyan
# email: liangxinyan48@163.com
import numpy as np
import random
import data_utils.config as config

import utils
import random_tree
import tree_to_strlist
import utils_tree


# split_data = config.get_configs()['split_data']

def generate_population(views=10, pop_size=10, verbose=0):  ##老师原来的程序
    '''
    种群初始化
    :param views: 视图个数
    :param pop_size: 种群大小
    :return:
    '''
    fusion_ways = config.get_configs()['fusion_ways']
    population = []
    population_set = set()

    """
    这里我需要修改
    """

    while len(population) < pop_size:
        # for i in range(pop_size):
        # view_code at least contains two elements
        view_code = random.sample(range(0, views), k=random.randint(2, views))
        # sample(seq, n) 从序列seq中选择n个随机且独立的元素；
        fusion_code = random.choices(range(0, len(fusion_ways)), k=len(view_code) - 1)
        pop = view_code + fusion_code
        if verbose == 1:
            print(f'view_code:{view_code}')
            print(f'fusion_code:{fusion_code}')
            print(f'pop:{pop}')
            print('=' * 30)
        if utils.list2str(pop) not in population_set:
            population.append(pop)
            population_set.add(utils.list2str(pop))  ## 如果已经存在就不放进去 防止重复
    return population


def generate_population_tree(views=5, pop_size=10, verbose=1):  ##老师原来的程序，生成初始种群
    '''
        种群初始化
        :param views: 视图个数
        :param pop_size: 种群大小
        :return:
    '''
    fusion_ways = config.get_configs()['fusion_ways']  ## 拿到五种融合方式
    population = []  ## 开始的初始化
    population_set_tree = set()  ## 弄一个如果你存在过的种群就别出现了
    numbers = [0, 1, 2, 3, 4]
    top_three_operators = [0, 1, 2, 3, 4]

    while len(population) < pop_size:
        view_code = random.choices(numbers, k=random.randint(2, len(numbers)))
        fusion_code = random.choices(top_three_operators, k=len(view_code) - 1)
        # split_code = random.choice(split_data)
        pop_tree = random_tree.randomTree(view_code, fusion_code)
        pop = utils_tree.tree_to_list2(pop_tree)
        # pop_split = pop.append(str(split_code))
        if verbose == 1:
            print(f'view_code:{view_code}')
            print(f'fusion_code:{fusion_code}')
            print(f'pop:{pop}')
            print('=' * 30)
        if tree_to_strlist.tree_list2str(pop) not in population_set_tree:
            population.append(pop)
            population_set_tree.add(utils.list2str(pop))  ## 如果已经存在就不放进去 防止重复
    return population


if __name__ == '__main__':
    population = generate_population_tree()
    # for i in population:
    #     print(i)
