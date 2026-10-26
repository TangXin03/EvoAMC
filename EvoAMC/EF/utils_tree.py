from  treelib import  Tree,Node
import random
import data_utils.config as config
import random_tree
import utils
import random
"""
主要是一些转化函数
"""
## 存一些全局参数
def tree_to_list2(tree):  ## 完成没问题   一棵树 转化为 一个树融合序列
    iter_tree = tree.expand_tree()
    listhead = []
    for i in iter_tree:
        listhead.append(tree.get_node(i))
    lists = [i.tag for i in listhead]
    lists = list(reversed(lists))
    return lists

def new_tree_1(top_five_views,top_tree_operators): ## 随机得到一棵树
    #fusion_ways = config.get_configs()['fusion_ways']
    #views = config.get_configs()['nb_view']
    view_code = random.sample(top_five_views, k=random.randint(2, len(top_five_views)))
    fusion_code = random.choices(top_tree_operators, k=len(view_code) - 1)
    pop_tree = random_tree.randomTree(view_code, fusion_code)
    #print("打印一下新生成的树")
    #print(pop_tree)
    return pop_tree

def new_tree_2(top_five_views,top_tree_operators): ## 随机得到一棵树
    #fusion_ways = config.get_configs()['fusion_ways']
    #views = config.get_configs()['nb_view']
    view_code = random.choices(top_five_views, k=random.randint(2, len(top_five_views)))
    fusion_code = random.choices(top_tree_operators, k=len(view_code) - 1)
    pop_tree = random_tree.randomTree(view_code, fusion_code)
    #print("打印一下新生成的树")
    #print(pop_tree)
    return pop_tree




def list_to_tree(strtrees):  ##  一个融合树序列转化为一棵树展示 用于后期的变异 交叉 生成下一代
    idx = utils.idxx
    stackstree = []  ## 存放所有子树的栈
    k = 0  ## 标记树根位置
    for treenode in strtrees:
        k += 1
        node = treenode
        tree = Tree()
        if (node[0] != '-'):
            tree.create_node(tag=treenode, identifier=idx)
            stackstree.append(tree)
        else:
            if (k != len(strtrees)):
                tree.create_node(tag=treenode, identifier=idx)  ## 创造一个子树
            else:
                tree.create_node(tag=treenode, identifier=idx)
            tree.paste(idx, stackstree[-1])
            stackstree.pop()
            tree.paste(idx, stackstree[-1])
            stackstree.pop()
            stackstree.append(tree)
        idx = idx + 1
    treepop = stackstree[0]
    utils.idxx = idx
    return treepop

def tree_list2str(list1):  ## list 转化为字符串 用于判重  减用来代替了融合运算符号 这里我选择用加号
    return '+'.join([str(i) for i in list1])


def viewfusion(liststr): ##  这个就是还原融合树序列  可以把视图 和 融合算子分开  但是 前后顺序关系是不变的
    views = []
    fusions = []
    viewnum = 0
    for i in liststr:
        if(i[0] != '-'):
            views.append(int(i))
            viewnum +=1
        else:
            fusions.append(int(i[1]))
    view_fusion_code = views + fusions
    return view_fusion_code,viewnum  ## 返回整个序列  和 视图多少

def new_tree(): ## 随机得到一棵树
    fusion_ways = config.get_configs()['fusion_ways']
    views = config.get_configs()['nb_view']
    view_code = random.sample(range(0, views), k=random.randint(2, views))
    fusion_code = random.choices(range(0, len(fusion_ways)), k=len(view_code) - 1)
    pop_tree = random_tree.randomTree(view_code, fusion_code)
    print("打印一下新生成的树")
    print(pop_tree)
    return pop_tree



## 这是我写的新的划分数据集的函数

def split(full_list,shuffle = False ,ratio = 0.2):
    """
    返回一个切分的数据集呀
    出错了再说
    :param full_list:
    :param full_y:
    :param shuffle:
    :param ratio:
    :return:
    """
    n_toal = len(full_list)
    offset = int(n_toal * ratio)
    print("我要划分多少",offset)
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    return sublist_1

