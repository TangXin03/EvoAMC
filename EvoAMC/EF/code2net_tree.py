# encoding=utf-8
# author: liang xinyan
# email: liangxinyan48@163.com
import torch

import data_utils.config as config
import utils_tree
from torch import nn

paras = config.get_configs()
fusion_ways = paras['fusion_ways']
fused_nb_feats = paras['fused_nb_feats']
classes = paras['classes']
paras.setdefault('fusion_L', 1)
paras.setdefault('fusion_C', 1)
fusion_L = paras['fusion_L']  ## 融合长度
fusion_C = paras['fusion_C']  ## 融合通道数
from torch.nn import functional as F
## 导入新的融合维度
import math

idx = 0  ## 树的ID地址


def sign_sqrt(x):
    """计算符号平方根"""
    eps = 1e-10
    return torch.sign(x) * torch.sqrt(torch.abs(x) + eps)


def l2_norm(x):
    """计算L2范数"""
    return torch.nn.functional.normalize(x, p=2, dim=-1)


def fusion(x1, x2, way='add'):
    """融合两个特征张量"""
    if way == 'add':
        fusion_x = torch.add(x1, x2)
    elif way == 'multiply':
        fusion_x = torch.mul(x1, x2)
    elif way == 'concatenate':
        fusion_x = torch.cat((x1, x2), dim=-1)
        linear_layer = torch.nn.Linear(fusion_x.shape[1], fused_nb_feats).to(fusion_x.device)
        fusion_x = linear_layer(fusion_x)
    elif way == 'maximum':
        fusion_x = torch.max(x1, x2)
    elif way == 'average':
        fusion_x = torch.div(torch.add(x1, x2), 2.0)

    # print("我已经融合了", fusion_x)
    return fusion_x


class ScaledDotAttn1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.ln = nn.LayerNorm(128)  # 1D LayerNorm

    def forward(self, x1, x2):
        q = x1.transpose(0, 1)  # Transpose for 1D data
        k = x2
        v = x2.transpose(0, 1)  # Transpose for 1D data

        d_k = q.size(-1)
        scores = torch.matmul(q, k) / math.sqrt(d_k)

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(0, 1)  # Transpose back
        out = self.dropout(out)
        out = self.ln(out)

        return out


def code2net_tree(individual_code, nb_feats=[1024, 2048, 1028], listtree=''):
    individual_code, nb_view = utils_tree.viewfusion(listtree)
    # print("随机生成树融合编码的序列和视图大小", individual_code, nb_view)

    # 生成nb_view网络输入
    input_x = []
    x_bn = []
    for i in range(nb_view):
        view_id = individual_code[i]  # 视图编号
        layer = nn.Sequential(
            nn.Linear(in_features=nb_feats[view_id], out_features=fused_nb_feats),
            nn.BatchNorm1d(num_features=fused_nb_feats),
            nn.ReLU()
        )
        input_x.append(layer)
        x_bn.append(nn.BatchNorm1d(num_features=fused_nb_feats))

    # 融合过程
    x = input_x.copy()
    if nb_view > 1:
        individual_code1, vsize = listtree, nb_view
        # print("我是什么", individual_code1, vsize)
        listview = []  ## 这个就是栈了
        for index, i in enumerate(individual_code1):
            if (i[0] != '-'):
                listview.append(index)
            else:
                e1 = listview[-1]
                listview.pop()
                e2 = listview[-1]
                listview.pop()
                f1 = int(i[1])
                fusion_x = fusion(x1=x[e1], x2=x[e2], way=fusion_ways[f1])
                x.append(fusion_x)
                listview.append(vsize)
                vsize += 1
        # print("最终结果", x[-1])

    # 标准化、正则化和输出层
    fusion_x = nn.Sequential(*x)(x_bn[-1])
    fusion_x = torch.sign(torch.sqrt(torch.abs(fusion_x)))
    fusion_x = torch.nn.functional.normalize(fusion_x, p=2, dim=1)

    out_x = nn.Linear(in_features=fused_nb_feats, out_features=classes)
    model = nn.Sequential(*input_x, *x, x_bn[-1], fusion_x, out_x)
    return model


class FusionModel(nn.Module):
    def __init__(self, individual_code, nb_feats, listtree=''):
        """
        :param individual_code:    这个是视图数 操作算子
        :param nb_feats:  视图原来的维度大小
        :param listtree:  这个是一个融合序列
        """
        super(FusionModel, self).__init__()
        self.individual_code, self.nb_views = utils_tree.viewfusion(listtree)  ## 返回一个个体序列是视图+融合算子、第二个参数是视图数量
        self.listtree = listtree
        self.nb_feats = nb_feats
        self.fusion_x = None
        self.view_models = nn.ModuleList()
        for i in range(self.nb_views):
            self.view_models.append(nn.Sequential(
                nn.Linear(nb_feats[i], fused_nb_feats),
                nn.BatchNorm1d(fused_nb_feats),
                nn.ReLU()
            ))
        ### 这个是最后的转化 但是这里 要进行融合的一步操作  fused_nb_feats * nb_views 不应该是这样子的  应该就是fused_nb_feats
        self.bn = nn.BatchNorm1d(fused_nb_feats)
        self.fc = nn.Linear(fused_nb_feats, classes)
        self.dropout1 = nn.Dropout(0.2)
        # self.sigmoid = nn.Sigmoid()
        # 注意力权重生成网络
        self.attn_layer = nn.Sequential(
            nn.Linear(fused_nb_feats * 2, fused_nb_feats),
            nn.ReLU(),
            nn.Linear(fused_nb_feats, 2),  # 为 x1 和 x2 生成注意力权重
            nn.Softmax(dim=-1)  # 归一化注意力
        )
        ## 这个是拼接起来之后需要模态对其的
        self.cat_fusion_layer = nn.Sequential(
            nn.Linear(fused_nb_feats * 2, fused_nb_feats),
            nn.BatchNorm1d(fused_nb_feats),
            nn.ReLU()
        )

    # def forward(self, x):
    #     views = [x[i] for i in range(len(self.view_models))]
    #     # print("我在哪里运行",views[2].device.type)  # 应该输出 "cuda"
    #     # print("views", views[0].shape, views[1].shape, views[2].shape)
    #     for i in range(len(views)):
    #         views[i] = views[i].to(torch.float32)
    #     x_processed = [self.view_models[i](views[i]) for i in range(len(self.view_models))]  # 通过对应的视图模型处理视图
    #     # print("x_processed", x_processed[0].shape)
    #     ### x_processed   里面就是一个list  里面有维度对齐之后的样子
    #     """
    #     这个应该是主要操作
    #     在这里进行融合
    #     """
    #     cnt = 0
    #     self.fusion_x = None
    #     if self.nb_views == 1:
    #         self.fusion_x = x_processed[0]  ## 如果视图等于1 那么它也只有一个
    #     else:
    #         individual_code1, vsize = self.listtree, self.nb_views
    #         listview = []  ## 这个就是栈了
    #         for index, i in enumerate(individual_code1):
    #             if (i[0] != '-'):
    #                 listview.append(cnt)
    #                 cnt += 1
    #             else:
    #                 e1 = listview[-1]
    #                 listview.pop()
    #                 e2 = listview[-1]
    #                 listview.pop()
    #                 f1 = int(i[1])
    #                 self.fusion_x = self.fusion(x1=x_processed[e1], x2=x_processed[e2], fused_nb_feats=fused_nb_feats,
    #                                             way=fusion_ways[f1])
    #                 x_processed.append(self.fusion_x)
    #                 # print("加入栈的新元素",fusion_x,vsize)
    #                 # print(x)
    #                 listview.append(vsize)
    #                 vsize += 1
    #     x_stacked = self.fusion_x
    #     # out = self.bn(x_stacked)
    #     out = sign_sqrt(x_stacked)
    #     out_x = self.fc(out)
    #     # out_x = self.dropout1(out_x) 加了之后效果奇差无比
    #     # out_x = self.sigmoid(out_x)
    #     return out_x
    def forward(self, x, return_feature=False):

        views = [x[i] for i in range(len(self.view_models))]

        for i in range(len(views)):
            views[i] = views[i].to(torch.float32)

        x_processed = [self.view_models[i](views[i]) for i in range(len(self.view_models))]

        cnt = 0
        self.fusion_x = None

        if self.nb_views == 1:
            self.fusion_x = x_processed[0]
        else:
            individual_code1, vsize = self.listtree, self.nb_views
            listview = []
            for index, i in enumerate(individual_code1):
                if (i[0] != '-'):
                    listview.append(cnt)
                    cnt += 1
                else:
                    e1 = listview[-1]
                    listview.pop()
                    e2 = listview[-1]
                    listview.pop()
                    f1 = int(i[1])

                    self.fusion_x = self.fusion(
                        x1=x_processed[e1],
                        x2=x_processed[e2],
                        fused_nb_feats=fused_nb_feats,
                        way=fusion_ways[f1]
                    )

                    x_processed.append(self.fusion_x)
                    listview.append(vsize)
                    vsize += 1

        x_stacked = self.fusion_x

        # ===== ⭐ 这里就是 feature =====
        feat = sign_sqrt(x_stacked)

        out_x = self.fc(feat)

        # ===== ⭐ DANN关键 =====
        if return_feature:
            return feat, out_x
        else:
            return out_x

    def fusion(self, x1, x2, fused_nb_feats, way='add'):
        """融合两个特征张量"""
        if way == 'add':
            self.fusion_x = torch.add(x1, x2)
        elif way == 'mul':
            self.fusion_x = torch.mul(x1, x2)
        elif way == 'cat':
            self.fusion_x = self.cat_fusion_layer(torch.cat((x1, x2), dim=-1))
        # elif way == 'attn':
        #     concat_x = torch.cat([x1, x2], dim=-1)
        #     attn_weights = self.attn_layer(concat_x)  # [B, 2]
        #     w1 = attn_weights[:, 0].unsqueeze(-1)
        #     w2 = attn_weights[:, 1].unsqueeze(-1)
        #     self.fusion_x = w1 * x1 + w2 * x2
        elif way == 'max':
            self.fusion_x = torch.max(x1, x2)
        elif way == 'avg':
            self.fusion_x = torch.div(torch.add(x1, x2), 2.0)
        else:

            raise ValueError(f"Unsupported fusion way: {way}")
        return self.fusion_x


class FusionModel_NTU(nn.Module):
    def __init__(self, individual_code, nb_feats, listtree=''):
        """

        :param individual_code:    这个是视图数 操作算子
        :param nb_feats:  视图原来的维度大小
        :param listtree:  这个是一个融合序列
        """
        super(FusionModel_NTU, self).__init__()
        self.individual_code, self.nb_views = utils_tree.viewfusion(listtree)  ## 返回一个个体序列是视图+融合算子、第二个参数是视图数量
        self.listtree = listtree
        self.nb_feats = nb_feats
        self.fusion_x = None
        self.fusion_L = fusion_L  ## 求的 H W  取的长度
        self.fusion_C = fusion_C  ## 通道数
        self.view_models = nn.ModuleList()
        self.pool = nn.AdaptiveAvgPool2d((self.fusion_L, 1))
        for i in range(self.nb_views):
            self.view_models.append(nn.Sequential(
                nn.Conv1d(self.nb_feats[i], self.fusion_C, 1, 1),  ### 一个维度卷积
                nn.Flatten(),
                nn.Linear(self.fusion_C * self.fusion_L, fused_nb_feats),
                nn.BatchNorm1d(fused_nb_feats),
                nn.ReLU()
            ))
        ### 这个是最后的转化 但是这里 要进行融合的一步操作  fused_nb_feats * nb_views 不应该是这样子的  应该就是fused_nb_feats
        self.bn = nn.BatchNorm1d(fused_nb_feats)
        self.fc = nn.Linear(fused_nb_feats, classes)
        self.dropout1 = nn.Dropout(0.2)
        # self.sigmoid = nn.Sigmoid()
        ## 这个是拼接起来之后需要模态对其的

    def forward(self, x):
        views = [x[i].unsqueeze(-1) for i in range(len(self.view_models))]
        mods = []
        ## 一系列操作以后L 已经是一样的 后面就是对通道进行对齐了  这样子做也是为了保证更多的信息
        for mod in views:
            mod = mod.view(mod.size(0), mod.size(1), mod.size(2), -1)
            mod = self.pool(mod)
            mod = mod.view(mod.size(0), mod.size(1), -1)
            # mod = F.interpolate(mod,self.fusion_L)
            mods.append(mod)
        # print("我在哪里运行",views[2].device.type)  # 应该输出 "cuda"
        x_processed = [self.view_models[i](mods[i]) for i in range(len(self.view_models))]
        ### x_processed   里面就是一个list  里面有维度对齐之后的样子
        """
        这个应该是主要操作 
        在这里进行融合 
        """
        self.fusion_x = None
        if self.nb_views == 1:
            self.fusion_x = x_processed[0]  ## 如果视图等于1 那么它也只有一个
        else:
            individual_code1, vsize = self.listtree, self.nb_views
            listview = []  ## 这个就是栈了
            for index, i in enumerate(individual_code1):
                if (i[0] != '-'):
                    listview.append(index)
                else:
                    e1 = listview[-1]
                    listview.pop()
                    e2 = listview[-1]
                    listview.pop()
                    f1 = int(i[1])
                    self.fusion_x = self.fusion(x1=x_processed[e1], x2=x_processed[e2],
                                                fused_nb_feats=fused_nb_feats, way=fusion_ways[f1])
                    x_processed.append(self.fusion_x)
                    # print("加入栈的新元素",fusion_x,vsize)
                    # print(x)
                    listview.append(vsize)
                    vsize += 1
        x_stacked = self.fusion_x
        out = self.bn(x_stacked)
        out = sign_sqrt(out)
        out_x = self.fc(out)
        # out_x = self.dropout1(out_x) 加了之后效果奇差无比
        # out_x = self.sigmoid(out_x)
        return out_x

    def fusion(self, x1, x2, fused_nb_feats, way='add'):
        """融合两个特征张量"""
        if way == 'add':
            self.fusion_x = torch.add(x1, x2)
        elif way == 'mul':
            self.fusion_x = torch.mul(x1, x2)
        elif way == 'cat':
            self.fusion_x = torch.cat((x1, x2), dim=-1)
            device_index = self.fusion_x.device.index
            linear_layer = torch.nn.Linear(self.fusion_x.shape[1], fused_nb_feats).to(
                device_index)
            self.fusion_x = linear_layer(self.fusion_x)
        elif way == 'max':
            self.fusion_x = torch.max(x1, x2)
        elif way == 'avg':
            self.fusion_x = torch.div(torch.add(x1, x2), 2.0)
        return self.fusion_x

    def fusion_NTU(self, x1, x2, fused_nb_feats, way='add'):
        """融合两个特征张量"""
        if way == 'add':
            self.fusion_x = torch.add(x1, x2)
        elif way == 'mul':
            self.fusion_x = torch.mul(x1, x2)
        elif way == 'cat':
            self.fusion_x = torch.cat((x1, x2), dim=-1)
            device_index = self.fusion_x.device.index
            linear_layer = torch.nn.Linear(self.fusion_x.shape[1], fused_nb_feats).to(
                device_index)  ## 细节 这个是最新创的 必须要弄到GPU上
            self.fusion_x = linear_layer(self.fusion_x)
        elif way == 'max':
            self.fusion_x = torch.max(x1, x2)
        elif way == 'avg':
            self.fusion_x = torch.div(torch.add(x1, x2), 2.0)
        return self.fusion_x
