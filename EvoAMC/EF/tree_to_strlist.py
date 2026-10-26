from treelib import Tree, Node
import utils


def tree_to_list(tree):  ## 原始的有一丝小问题 已经舍去
    listhead = list(reversed(tree.all_nodes()))
    lists = [i.tag for i in listhead]
    return lists


def tree_to_list2(tree):  ## 完成没问题   完全可以去转化
    iter_tree = tree.expand_tree()  ##深度
    listhead = []
    for i in iter_tree:
        listhead.append(tree.get_node(i))
    lists = [i.tag for i in listhead]
    lists = list(reversed(lists))
    return lists


def list_to_tree(strtrees):
    idx = utils.idxx
    stackstree = []  ## 存放所有子树的栈
    idxroot = 0  ## 树根是谁
    k = 0  ## 标记树根位置
    for treenode in strtrees:
        k += 1
        node = treenode
        tree = Tree()
        if (node[0] != '-'):
            ##  里面保存的是 idx  也就是唯一的便是
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
    # stackstree[0].show()
    treepop = stackstree[0]
    # print("我是真树")
    # treepop.show()
    # print(treepop.all_nodes())  ## 得到所有节点 是按照 根右左
    """
    order 后序
    """
    treepoporder = list(reversed(treepop.all_nodes()))
    utils.idxx = idx
    return treepop


def tree_list2str(list1):
    return '-'.join([str(i) for i in list1])


def viewfusion(liststr):  ## 里面的视图是按照顺序来的 比如有三个  0 1 2
    views = []
    fusions = []
    viewnum = 0
    for i in liststr:
        if (i[0] != '-'):
            views.append(int(i))
            viewnum += 1
        else:
            fusions.append(int(i[1]))
    view_fusion_code = views + fusions
    return view_fusion_code, viewnum
