import torch
import torch.nn as nn
from torch import tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F


class ATT(nn.Module):
    def __init__(self, feature_size=16, num_heads=1, dropout=0, bias=False):
        # 建议将feature_size设为num_heads的倍数
        super(ATT, self).__init__()
        self.att = MHATT(query_size=feature_size, key_size=feature_size,
                         value_size=feature_size, num_hiddens=feature_size,
                         num_heads=num_heads, dropout=dropout, bias=bias)

    def forward(self, target, source):
        return self.att(target, source, source)


class DPATT(nn.Module):
    """ Dot Product Attention """

    def __init__(self, dropout=0, **kwargs):
        super(DPATT, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        d = tensor(queries.shape[-1])
        alphas = F.softmax(queries @ keys.transpose(-2, -1) / torch.sqrt(d), dim=-1)
        h = self.dropout(alphas) @ values
        return h


class MHATT(nn.Module):
    """ Multi-Headed Dot Product Attention """

    def __init__(self, query_size, key_size, value_size, num_hiddens,
                 num_heads=1, dropout=0, bias=False, **kwargs):
        super(MHATT, self).__init__(**kwargs)

        self.num_heads = num_heads
        self.att = DPATT(dropout=dropout)

        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values):
        queries = MHATT.transpose_qkv(self.W_q(queries), self.num_heads)
        keys = MHATT.transpose_qkv(self.W_k(keys), self.num_heads)
        values = MHATT.transpose_qkv(self.W_v(values), self.num_heads)

        # output的形状:(batch_size*num_heads，查询的个数，num_hiddens/num_heads)
        output = self.att(queries, keys, values)

        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = MHATT.transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

    @staticmethod
    def transpose_qkv(X, num_heads):
        """为了多注意力头的并行计算而变换形状"""
        # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)

        # 输出X的形状:
        # (batch_size，查询或者“键－值”对的个数，num_heads，num_hiddens/num_heads)
        X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

        # 输出X的形状:
        # (batch_size，num_heads，查询或者“键－值”对的个数，num_hiddens/num_heads)
        X = X.permute(0, 2, 1, 3)

        # 最终输出的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，num_hiddens/num_heads)
        return X.reshape(-1, X.shape[2], X.shape[3])

    @staticmethod
    def transpose_output(X, num_heads):
        """逆转transpose_qkv函数的操作"""
        X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)


if __name__ == '__main__':
    batch_size, feature_size = 2, 4
    n, num_heads = 3, 2

    # 把target的小批量的t，reshape成(batch_size,1,fc维度)
    target_feature = torch.ones([batch_size, 1, feature_size])

    # 把source的s，reshape成(batch_size,n,fc维度)，
    # 其中n为‘源域数’或其他(根据需要而定)，n不会影响模型的参数大小
    source_features = torch.ones([batch_size, n, feature_size])

    # query_size = key_size = value_size = fc维度
    # num_hiddens = 输出向量的维度 = fc维度
    att = ATT(feature_size=feature_size, num_heads=num_heads)

    # queries来自target，keys和values来自source
    h = att(target_feature, source_features)  # (batch_size,1,fc维度)

    print()
