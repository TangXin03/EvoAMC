import logging
import os.path
import random
import copy
import utils_tree
import data_utils.config as config
from os.path import join as pjoin

paras = config.get_configs()
data_name = paras['data_name']
nb_fusion_way = len(paras['fusion_ways'])
nb_view = 3  # paras['nb_view']
is_remove = paras['is_remove']
## 暂时先用这个
import utils

result_save_dir = pjoin(
    paras['result_root'],
    data_name + '_view_result',
    paras['result_save_dir'],
)
LOGGER = logging.getLogger("amc_ef")


def _arch_key(individual):
    return '+'.join([str(i) for i in individual])


def _filter_scored_individuals(individuals, shared_code_acc, context):
    scored = []
    missing = []
    for individual in individuals:
        key = _arch_key(individual)
        if key in shared_code_acc:
            scored.append(individual)
        else:
            missing.append(key)

    if missing:
        LOGGER.warning(
            "%s | missing fitness records | count=%d | arches=%s",
            context,
            len(missing),
            missing,
        )
    return scored


def get_all_nodes_identifier(tree):  ## 得到所有节点  ##除了根节点不要
    nodes = tree.all_nodes()  ## 得到这棵树的所有节点
    identifiersfph = []
    for node in nodes:
        if (tree.parent(node.identifier) != None):
            identifiersfph.append(node.identifier)
        else:
            nodes.remove(node)
    return nodes, identifiersfph


def get_leaf_nodes_identifier(tree):  ## 得到所有叶子节点
    nodes = tree.leaves()
    identifiers = [node.identifier for node in nodes]
    return nodes, identifiers


def split_tree(tree, nid):  ## 分割
    tree_copy = copy.deepcopy(tree)
    removed_tree = tree_copy.remove_subtree(nid=nid, identifier=nid)  ##删除这棵树
    return tree_copy, removed_tree


def crossover(tree1, tree2, crossover_rate, is_remove=is_remove, max_deep=15):  ## 交叉
    ## 如果本身为空直接特判取消

    if len(tree1) == 0 or len(tree2) == 0 or len(tree1) == 1 or len(tree2) == 1:  ### 这里特判一下就好了
        return tree1, tree2
    # print("第一棵树大小",len(tree1))
    # print("第二棵树大小",len(tree2))

    r = random.random()  ## 生成一个交叉变异的概率
    if (r < crossover_rate):
        tree1_nodes, tree1_identifiers = get_all_nodes_identifier(tree1)  ## 得到所有节点的值
        tree2_nodes, tree2_identifiers = get_all_nodes_identifier(tree2)  ## 得到所有节点的值

        ## 这里会出现问题得特判一下
        if len(tree1) == 1:
            print("出现问题")
        tree1_split_point = random.choice(tree1_nodes)  ## 随机选择一颗子树
        tree2_split_point = random.choice(tree2_nodes)  ## 随机选择子树

        tree1_split_node = tree1_split_point
        tree2_split_node = tree2_split_point
        node = tree1.parent(tree1_split_node.identifier)
        if (node == None):
            tree1_split_node_parent = tree1_split_node.identifier
        else:
            tree1_split_node_parent = node.identifier
        node = tree2.parent(tree2_split_node.identifier)
        if (node == None):
            tree2_split_node_parent = tree2_split_node.identifier
        else:
            tree2_split_node_parent = node.identifier
        # print("找到了父母",tree1_split_node_parent, tree2_split_node_parent)
        tree1_left, tree1_right = split_tree(tree1, tree1_split_point.identifier)
        tree2_left, tree2_right = split_tree(tree2, tree2_split_point.identifier)
        tree1_left.paste(tree1_split_node_parent, tree2_right)
        tree2_left.paste(tree2_split_node_parent, tree1_right)
        ##我现在得到了树 我要去看看我的树是不是太深了 或者是 我是否要执行消除重复操作
        # print("交叉树未去重结果")
        # print(tree1_left)
        # print(tree2_left)
        if is_remove == True:
            tree1_left = quchong(tree1_left)
            tree2_left = quchong(tree2_left)
        if tree1_left.depth() > max_deep:
            tree1_left = quchong(tree1_left)
        if tree2_left.depth() > max_deep:
            tree2_left = quchong(tree2_left)

        return tree1_left, tree2_left  ## 返回交叉变异的树
    else:
        if is_remove:
            tree1 = quchong(tree1)
            tree2 = quchong(tree2)
        if tree1.depth() > max_deep:
            tree1 = quchong(tree1)
            tree2 = quchong(tree2)
        return tree1, tree2


def get_branch_nodes_identifier(tree):
    all_nodes = tree.all_nodes()
    branch_nodes = []
    identifiersfph = []
    for node in all_nodes[:]:
        if len(tree.is_branch(node.identifier)) != 0 and tree.parent(node.identifier) is not None:  # 判断是否为叶子节点
            branch_nodes.append(node)
            identifiersfph.append(node.identifier)
        else:
            all_nodes.remove(node)
    return branch_nodes, identifiersfph


def crossover_add(tree1, tree2, crossover_rate, is_remove=is_remove, max_deep=15):  ## 交叉

    if len(tree1) == 0 or len(tree2) == 0 or len(tree1) == 1 or len(tree2) == 1:  ### 这里特判一下就好了
        return tree1, tree2
    # print("第一棵树大小",len(tree1))
    # print("第二棵树大小",len(tree2))
    r = random.random()  ## 生成一个交叉变异的概率
    if (r < crossover_rate):
        tree1_nodes, tree1_identifiers = get_leaf_nodes_identifier(tree1)  ## 第一颗是主树 所以我们只用获取叶子节点
        tree2_nodes, tree2_identifiers = get_branch_nodes_identifier(tree2)  ## 得到所有的分支节点

        if len(tree2_nodes) == 0:
            tree2_nodes, tree2_identifiers = get_all_nodes_identifier(tree2)

        if len(tree1) == 1:
            print("出现问题")

        tree1_split_point = random.choice(tree1_nodes)  ## 随机选择一颗子树
        tree2_split_point = random.choice(tree2_nodes)  ## 随机选择子树

        tree1_split_node = tree1_split_point
        tree2_split_node = tree2_split_point
        node = tree1.parent(tree1_split_node.identifier)
        if (node == None):
            tree1_split_node_parent = tree1_split_node.identifier
        else:
            tree1_split_node_parent = node.identifier

        if (node == None):
            tree2_split_node_parent = tree2_split_node.identifier
        else:
            tree2_split_node_parent = node.identifier
        tree2_left, tree2_right = split_tree(tree2, tree2_split_point.identifier)

        # print(tree2_left)
        tree1.paste(tree1_split_node_parent, tree2_left)

        if is_remove == True:
            tree1 = quchong(tree1)

        if tree1.depth() > max_deep:
            tree1 = quchong(tree1)
        return tree1, tree1
    else:
        if is_remove:
            tree1 = quchong(tree1)
            tree2 = quchong(tree2)
        if tree1.depth() > max_deep:
            tree1 = quchong(tree1)
            tree2 = quchong(tree2)
        return tree1, tree2


def mutation(tree, mutation_rate, is_remove=is_remove, max_deep=15):  ## 这个是变异某个节点
    nodes = tree.all_nodes()  ## 随机得到一些点 就可以进行点的变异
    ## 接下来我应该去随机变异 一些节点
    node = random.choice(nodes)
    idtag = node.tag  ## 随机选择
    print("选中变异树节点", idtag, type(idtag))
    r = random.random()
    if (r < mutation_rate):
        if (idtag[0] == '-'):  # 说明应该是变异应该是融合算子
            mutation_view = list(range(nb_fusion_way))
            # print(mutation_view)
            id = random.choice(mutation_view)
            idtag = '-' + str(id)
        else:
            mutation_view = list(range(nb_view))
            # print(mutation_view)
            id = random.choice(mutation_view)
            idtag = str(id)
            # print("选中的",id)
        node.tag = idtag  ## 返回过去
        if is_remove:
            tree = quchong(tree)
    else:
        if is_remove:
            tree = quchong(tree)

    if (tree.depth() > max_deep):
        tree = quchong(tree)
    return tree


def mutation_new_tree_crossover(tree, mutation_rate, is_remove=is_remove, max_deep=15,
                                flat=0):  ## 现在我收到了 一棵树 那我要做什么呢  要随机去变异节点的值
    r = random.random()
    if (r < mutation_rate):
        # tree_mut = utils_tree.new_tree() ## 抽出来一颗新树 用于变异

        # tree1,tree2 = crossover(tree,tree_mut,1,is_remove,15)  ##交叉一下  只要取第一颗就好了

        # print("False")
        # if flat == 0:
        #     tree_mut = utils_tree.new_tree_1([0, 1, 2], [0, 1, 3, 4])  ## 抽出来一颗新树 用于变异
        #     tree1, tree2 = crossover(tree, tree_mut, 1, is_remove, 15)  ##交叉一下  只要取第一颗就好了
        # else:
        #     tree_mut = utils_tree.new_tree_2([0, 1, 2], [0, 1, 3, 4])  ## 抽出来一颗新树 用于变异
        #     tree1, tree2 = crossover_add(tree, tree_mut, 1, is_remove, 15)  ##交叉一下  只要取第一颗就好

        if flat == 0:
            tree_mut = utils_tree.new_tree_1([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])  ## 抽出来一颗新树 用于变异
            tree1, tree2 = crossover(tree, tree_mut, 1, is_remove, 15)  ##交叉一下  只要取第一颗就好了
        else:
            tree_mut = utils_tree.new_tree_2([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])  ## 抽出来一颗新树 用于变异
            tree1, tree2 = crossover_add(tree, tree_mut, 1, is_remove, 15)  ##交叉一下  只要取第一颗就好
        # tree_mut = utils_tree.new_tree_2([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])  ## 抽出来一颗新树 用于变异
        # tree1, tree2 = crossover_add(tree, tree_mut, 1, is_remove, 15)  ##交叉一下  只要取第一颗就好

        if is_remove:
            tree1 = quchong(tree1)
        if tree1.depth() > max_deep:
            tree1 = quchong(tree1)
        flat = True
        return tree1, flat
    else:
        if is_remove:
            tree = quchong(tree)
        if tree.depth() > max_deep:
            tree = quchong(tree)
        flat = False
        return tree, flat


def quchong(tree_p):
    list_tree = utils_tree.tree_to_list2(tree_p)
    quchong_tree = []  ## 去重之后的树融合序列
    """
    这里可以模拟出来 用融合方式
    """
    num_views = 0
    for i in list_tree:
        if i[0] != '-':
            if i not in quchong_tree:
                quchong_tree.append(i)
                num_views += 1
        else:
            if num_views >= 2:
                quchong_tree.append(i)
                num_views -= 1
    quchongtree = utils_tree.list_to_tree(quchong_tree)
    return quchongtree  ## 完成去重序列


def gen_offspring_tra_n(P_t, num_offspring):  ## 种群的的里面个体编码变了  l,生成种群后代,返回n/n+1棵树
    shared_code_acc = utils.load_result_n()  ## 这是一个字典

    Fu = []

    # print(shared_code_acc)
    # 1. Crossover
    def select_p():  ## 两个里面选一个好的
        two = random.sample(range(len(P_t)), 2)  ## 随机选两个  二元竞标赛 就是这个意思
        a1 = '+'.join([str(i) for i in P_t[two[0]]])
        a2 = '+'.join([str(i) for i in P_t[two[1]]])
        p1 = P_t[two[0]] if shared_code_acc[a1] > shared_code_acc[a2] else P_t[two[1]]  ## 两个里面选一个精度好的返回
        return p1

    Q_t = []
    while len(Q_t) < len(P_t):
        p1 = select_p()
        p2 = select_p()
        while '+'.join(str(i) for i in p1) == '+'.join(str(i) for i in p2):  ## 防止重复
            p2 = select_p()
        p1_tree = utils_tree.list_to_tree(p1)
        p2_tree = utils_tree.list_to_tree(p2)

        for _ in range(num_offspring):  ## 修改：每对父代生成num_offspring个后代
            o1_tree, o2_tree = crossover(tree1=p1_tree, tree2=p2_tree, crossover_rate=paras['crossover_rate'])
            o1 = utils_tree.tree_to_list2(o1_tree)
            o2 = utils_tree.tree_to_list2(o2_tree)
            Q_t.append(o1)
            Q_t.append(o2)

    # 2. Mutation
    Q_tt = []
    for p in Q_t:
        p_tree = utils_tree.list_to_tree(p)

        # p1_tree = mutation_new_tree_crossover(p_tree,mutation_rate=paras['mutation_rate'])
        p1_tree = mutation(p_tree, mutation_rate=paras['mutation_rate'], is_remove=is_remove, max_deep=15)

        p1 = utils_tree.tree_to_list2(p1_tree)
        Q_tt.append(p1)
    Q_t = Q_tt
    return Q_t


def gen_offspring(P_t, i_iter):  ## 种群的的里面个体编码变了  l,生成种群后代,返回n/n+1棵树
    shared_code_acc = utils.load_result()  ## 这是一个字典

    # print(shared_code_acc)
    # 1. Crossover
    def select_p():  ## 两个里面选一个好的
        two = random.sample(range(len(P_t)), 2)  ## 随机选两个  二元竞标赛 就是这个意思
        a1 = '+'.join([str(i) for i in P_t[two[0]]])
        a2 = '+'.join([str(i) for i in P_t[two[1]]])
        p1 = P_t[two[0]] if shared_code_acc[a1] > shared_code_acc[a2] else P_t[two[1]]  ## 两个里面选一个精度好的返回
        return p1

    Q_t = []
    while len(Q_t) < len(P_t):
        p1 = select_p()
        p2 = select_p()
        while '+'.join(str(i) for i in p1) == '+'.join(str(i) for i in p2):  ## 防止重复
            p2 = select_p()
        p1_tree = utils_tree.list_to_tree(p1)
        p2_tree = utils_tree.list_to_tree(p2)

        if i_iter >= 6:
            o1_tree, o2_tree = crossover(tree1=p1_tree, tree2=p2_tree,
                                         crossover_rate=paras['crossover_rate'] / 9)  # l,迭代次数较大时，减少交叉率（避免频繁结构重组）
        else:
            o1_tree, o2_tree = crossover(tree1=p1_tree, tree2=p2_tree, crossover_rate=paras['crossover_rate'])

        # o1_tree, o2_tree = crossover(tree1=p1_tree, tree2=p2_tree, crossover_rate=paras['crossover_rate'])

        o1 = utils_tree.tree_to_list2(o1_tree)
        o2 = utils_tree.tree_to_list2(o2_tree)
        Q_t.append(o1)
        Q_t.append(o2)
    # 2. Mutation
    Q_tt = []
    for p in Q_t:
        p_tree = utils_tree.list_to_tree(p)

        if i_iter >= 6:
            p1_tree, flat = mutation_new_tree_crossover(p_tree, mutation_rate=paras['mutation_rate'] * 4,
                                                        flat=1)  # l,迭代次数较大时，增加变异率以避免陷入局部最优
        else:
            p1_tree, flat = mutation_new_tree_crossover(p_tree, mutation_rate=paras['mutation_rate'], flat=0)

        # p1_tree = mutation_new_tree_crossover(p_tree,mutation_rate=paras['mutation_rate'])

        p1 = utils_tree.tree_to_list2(p1_tree)
        Q_tt.append(p1)
    Q_t = Q_tt
    return Q_t


def gen_offspring_n(P_t, i_iter, num_offspring=10):  ## 种群的的里面个体编码变了  l,生成种群后代,返回n/n+1棵树
    shared_code_acc = utils.load_result_n()  ## 这是一个字典

    # print(shared_code_acc)
    # 1. Crossover
    def select_p():  ## 两个里面选一个好的
        two = random.sample(range(len(P_t)), 2)  ## 随机选两个  二元竞标赛 就是这个意思
        a1 = '+'.join([str(i) for i in P_t[two[0]]])
        a2 = '+'.join([str(i) for i in P_t[two[1]]])
        p1 = P_t[two[0]] if shared_code_acc[a1] > shared_code_acc[a2] else P_t[two[1]]  ## 两个里面选一个精度好的返回
        return p1

    Q_t = []
    while len(Q_t) < len(P_t) * num_offspring:
        p1 = select_p()
        p2 = select_p()
        while '+'.join(str(i) for i in p1) == '+'.join(str(i) for i in p2):  ## 防止重复
            p2 = select_p()
        p1_tree = utils_tree.list_to_tree(p1)
        p2_tree = utils_tree.list_to_tree(p2)

        # 生成多个后代
        for _ in range(num_offspring):  ## 修改：每对父代生成num_offspring个后代
            if i_iter >= 6:
                o1_tree, o2_tree = crossover(tree1=p1_tree, tree2=p2_tree,
                                             crossover_rate=paras['crossover_rate'] / 9)  # l,迭代次数较大时，减少交叉率（避免频繁结构重组）
            else:
                o1_tree, o2_tree = crossover(tree1=p1_tree, tree2=p2_tree, crossover_rate=paras['crossover_rate'])
            o1 = utils_tree.tree_to_list2(o1_tree)
            o2 = utils_tree.tree_to_list2(o2_tree)
            Q_t.append(o1)
            Q_t.append(o2)
    # 2. Mutation
    Q_tt = []
    for p in Q_t:
        p_tree = utils_tree.list_to_tree(p)

        if i_iter >= 6:
            p1_tree, flat = mutation_new_tree_crossover(p_tree, mutation_rate=paras['mutation_rate'] * 4,
                                                        flat=1)  # l,迭代次数较大时，增加变异率以避免陷入局部最优
        else:
            p1_tree, flat = mutation_new_tree_crossover(p_tree, mutation_rate=paras['mutation_rate'], flat=0)

        p1 = utils_tree.tree_to_list2(p1_tree)
        Q_tt.append(p1)

    Q_t = Q_tt
    return Q_t


def gen_offspring_new(P_t, i_iter, num_offspring=10, result_fn=None):  ## 种群的的里面个体编码变了  l,生成种群后代,返回n/n+1棵树
    shared_code_acc = utils.load_result_n(result_fn) if result_fn is not None else utils.load_result_n()  # 加载个体的评估结果
    Fu = []

    unique_parent_map = {}
    for parent in P_t:
        unique_parent_map.setdefault('+'.join([str(i) for i in parent]), parent)
    parent_pool = list(unique_parent_map.values())
    if len(parent_pool) < 2:
        logging.getLogger("amc_ef_convergence").warning(
            "offspring generation | only %d unique parent(s), allowing self-pair to avoid dead loop",
            len(parent_pool),
        )
        parent_pool = P_t

    # print(shared_code_acc)
    # 1. Crossover
    def select_p():  ## 两个里面选一个好的
        if len(parent_pool) < 2:
            return parent_pool[0]
        two = random.sample(range(len(parent_pool)), 2)  ## 随机选两个  二元竞标赛 就是这个意思
        a1 = '+'.join([str(i) for i in parent_pool[two[0]]])
        a2 = '+'.join([str(i) for i in parent_pool[two[1]]])

        p1 = parent_pool[two[0]] if shared_code_acc[a1] > shared_code_acc[a2] else parent_pool[two[1]]  ## 两个里面选一个精度好的返回
        return p1

    # 1. 随机选择两个父代
    p1 = select_p()
    p2 = select_p()
    if len(parent_pool) >= 2:
        for _ in range(50):
            if '+'.join(str(i) for i in p1) != '+'.join(str(i) for i in p2):  # 确保父代不同
                break
            p2 = select_p()
        if '+'.join(str(i) for i in p1) == '+'.join(str(i) for i in p2):
            alternatives = [parent for parent in parent_pool if '+'.join(str(i) for i in parent) != '+'.join(str(i) for i in p1)]
            if alternatives:
                p2 = random.choice(alternatives)

    p1_tree = utils_tree.list_to_tree(p1)
    p2_tree = utils_tree.list_to_tree(p2)

    Fu.append(p1)
    Fu.append(p2)

    # 2. 生成多个子代，交叉
    offspring = []
    for _ in range(num_offspring):
        if i_iter >= 6:
            o1_tree, o2_tree = crossover(tree1=p1_tree, tree2=p2_tree,
                                         crossover_rate=paras['crossover_rate'] / 9)  # 迭代次数较大时，减少交叉率
        else:
            o1_tree, o2_tree = crossover(tree1=p1_tree, tree2=p2_tree, crossover_rate=paras['crossover_rate'])
        o1 = utils_tree.tree_to_list2(o1_tree)
        o2 = utils_tree.tree_to_list2(o2_tree)
        offspring.append(o1)
        offspring.append(o2)

    # 3. 变异
    O_spring = []
    for o in offspring:
        o_tree = utils_tree.list_to_tree(o)
        # if i_iter >= 6:
        #     o11_tree, _ = mutation_new_tree_crossover(o_tree, mutation_rate=paras['mutation_rate'] * 4,
        #                                               flat=1)  # 增加变异率            #####这儿有问题，记得改了以后再运行一次
        # else:
        #     o11_tree, _ = mutation_new_tree_crossover(o_tree, mutation_rate=paras['mutation_rate'], flat=0)
        # o11 = utils_tree.tree_to_list2(o11_tree)
        if i_iter >= 6:
            o11_tree, _ = mutation_new_tree_crossover(o_tree, mutation_rate=paras['mutation_rate'] * 4, flat=1)  # 增加变异率
        else:
            o11_tree, _ = mutation_new_tree_crossover(o_tree, mutation_rate=paras['mutation_rate'], flat=0)
        o11 = utils_tree.tree_to_list2(o11_tree)
        O_spring.append(o11)

    return Fu, O_spring


def selection(P_t, Q_t, result_fn=None):
    shared_code_acc = utils.load_result_n(result_fn) if result_fn is not None else utils.load_result_n()

    def select_p1(select_pool):  ## 一个选择池里面
        two = random.sample(range(len(select_pool)), 2)
        a1 = '+'.join([str(i) for i in select_pool[two[0]]])
        a2 = '+'.join([str(i) for i in select_pool[two[1]]])
        p1 = select_pool[two[0]] if shared_code_acc[a1] > shared_code_acc[a2] else select_pool[two[1]]
        return p1

    P_t1 = []
    Pt_Qt = P_t + Q_t
    while len(P_t1) < len(P_t):
        p = select_p1(Pt_Qt)
        P_t1.append(p)

    # 如果最好的个体不在P_t1，用最好的替换最差的
    max_code = []
    max_code_str = ''  ## 应该是这个空字符串
    min_code_str = ''
    for k, v in shared_code_acc.items():
        if v == max(shared_code_acc.values()):
            max_code_str = k  ##保存一下这个 我觉得我可以都用加号
            max_code = k.strip().split('+')
            # max_code = [int(i) for i in max_code] ### 这个转化我应该不用
        if v == min(shared_code_acc.values()):
            min_code_str = k
    is_max = False
    for i, v in enumerate(P_t1):
        v_str = utils_tree.tree_list2str(v)
        if v_str == max_code_str:
            is_max = True
            break
    if not is_max:
        min_i = 0
        for i, v in enumerate(P_t1):
            v_str = utils_tree.tree_list2str(v)
            if v_str == min_code_str:
                min_i = i
                break
        P_t1[min_i] = max_code
    return P_t1


def selection_new(P_t, Q_t):
    P_t1 = []
    shared_code_acc = utils.load_result_n()

    print(f'F_t: {P_t}')
    print(f'Q_t: {Q_t}')
    print(f'f: {shared_code_acc}')
    # 合并种群
    Pt_Qt = P_t + Q_t

    # 将个体转换为字符串键（用于 shared_code_acc）
    def to_key(individual):
        return '+'.join(map(str, individual))

    # 按分数从高到低排序
    sorted_population = sorted(Pt_Qt, key=lambda x: shared_code_acc[to_key(x)], reverse=True)
    P_t1 = sorted_population[:len(P_t)]
    return P_t1


def selection_new_2(P_t, Q_t):
    P_t1 = []
    shared_code_acc = utils.load_result_n()
    # 合并种群
    Pt_Qt = P_t + Q_t

    # 将个体转换为字符串键（用于 shared_code_acc）
    def to_key(individual):
        return '+'.join(map(str, individual))

    # 按分数从高到低排序
    sorted_population = sorted(Pt_Qt, key=lambda x: shared_code_acc[to_key(x)], reverse=True)

    # 选择不重复的个体
    selected_individuals = set()
    for individual in sorted_population:
        individual_key = to_key(individual)
        if individual_key not in selected_individuals:
            P_t1.append(individual)
            selected_individuals.add(individual_key)
            if len(P_t1) == len(P_t):
                break

    print("我到这儿了")
    with open(os.path.join(result_save_dir, 'best_individual.csv'), 'a+') as f:
        for individual in P_t1:
            key = to_key(individual)
            score = shared_code_acc[key]
            individual_str = '+'.join(map(str, individual))
            f.write(f'"{individual_str}",{score}\n')
    return P_t1
