import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# from .sync_batchnorm import BatchNorm1d, BatchNorm2d

BatchNorm1d = nn.BatchNorm1d
BatchNorm2d = nn.BatchNorm2d


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    *** Cosine similarity adjcent matrix
    """

    def __init__(self, in_features, out_features, norm_layer=BatchNorm1d, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features), requires_grad=True)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.bn = norm_layer(out_features)
        # nn.init.constant_(self.bn.weight, 0)
        # nn.init.constant_(self.bn.bias, 0)

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj_mask=None):
        adj = self.softmax(torch.matmul(input, input.permute(0, 2, 1)))
        if adj_mask is not None:
            adj = adj * adj_mask
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            output = output + self.bias
        output = output.permute(0, 2, 1)
        output = self.relu(output)
        output = self.bn(output)
        return output
        # if self.bias is not None:
        #     return self.bn(self.relu(output + self.bias))
        # else:
        #     return self.bn(self.relu(output))

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
