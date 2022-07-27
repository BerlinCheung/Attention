import torch
import torch.nn as nn
from torch import tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F


class DPATT(nn.Module):
    """ Dot Product Attention """

    def __init__(self, output_mode=0, dropout=0, **kwargs):
        """
        output_mode: 用于选择输出的模式，0为输出加权求和的结果，1为输出加权不求和的结果
        """
        super(DPATT, self).__init__(**kwargs)
        self.output_mode = output_mode
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        d = tensor(queries.shape[-1])
        # alphas: (batch_size, 1, n)
        alphas = F.softmax(queries @ keys.transpose(-2, -1) / torch.sqrt(d), dim=-1)
        if self.output_mode == 0:
            # h: (batch_size, 1, fc_size), 加权求和
            h = self.dropout(alphas) @ values
        else:
            alphas = self.dropout(alphas)
            # h: (batch_size, n, fc_size)，加权不求和
            h = values
            for i in range(alphas.shape[0]):
                for j in range(alphas.shape[-1]):
                    h[i, j, :] *= alphas[i, 0, j]
        return h, alphas


class ATT(nn.Module):
    def __init__(self, feature_size=16, qk_size=16,
                 use_source=True, is_init=True,
                 output_mode=0, dropout=0, bias=False):
        """
        feature_size: fc_size
        use_source: 是否使用source本身作为values
        is_init: 是否初始化W权重
        output_mode: 用于选择输出的模式，0为输出加权求和的结果，1为输出加权不求和的结果
        qk_size: 随便调（会影响参数大小）
        """
        super(ATT, self).__init__()
        self.att = DPATT(output_mode=output_mode, dropout=dropout)
        self.use_source = use_source

        self.W_q = nn.Linear(feature_size, qk_size, bias=bias)
        self.W_k = nn.Linear(feature_size, qk_size, bias=bias)
        self.W_v = nn.Linear(feature_size, feature_size, bias=bias)

        if is_init:
            self.W_q.weight = Parameter(F.softmax(torch.ones(size=(qk_size, feature_size),
                                                             requires_grad=True), dim=1))
            self.W_k.weight = Parameter(F.softmax(torch.ones(size=(qk_size, feature_size),
                                                             requires_grad=True), dim=1))
            self.W_v.weight = Parameter(F.softmax(torch.ones(size=(feature_size, feature_size),
                                                             requires_grad=True), dim=1))


    def forward(self, target, source):
        queries, keys = self.W_q(target), self.W_k(source)
        if self.use_source:
            values = source
        else:
            values = self.W_v(source)
        return self.att(queries, keys, values)


if __name__ == '__main__':
    batch_size, feature_size = 2, 4
    n = 3

    # 把target的小批量的t，reshape成(batch_size, 1, fc维度)
    target_feature = torch.ones([batch_size, 1, feature_size])

    # 把source的s，reshape成(batch_size, n, fc维度)，
    # 其中n为‘源域数’或其他(根据需要而定)，n不会影响模型的参数大小
    source_features = torch.ones([batch_size, n, feature_size])

    # feature_size = fc维度, qk_size可以调
    att = ATT(feature_size=feature_size, qk_size=feature_size,
              use_source=True, is_init=True, output_mode=0)

    # queries来自target，keys和values来自source
    # h: (batch_size,1,fc维度) / (batch_size,n,fc维度)
    # alphas: (batch_size, 1, n)
    h, alphas = att(target_feature, source_features)

    print()
