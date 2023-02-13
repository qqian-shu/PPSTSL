import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.graph_conv import calculate_laplacian_with_self_loop
import numpy as np

class GCN(nn.Module):
    def __init__(self, adj, input_dim: int, output_dim: int, **kwargs):
        super(GCN, self).__init__()
        # self.register_buffer(
        #     "laplacian", calculate_laplacian_with_self_loop(torch.FloatTensor(adj))
        # )
        self.adj = adj
        self.supports = 0
        self._num_nodes = adj.shape[0]
        self._input_dim = input_dim  # seq_len for prediction
        self._output_dim = output_dim  # hidden_dim for prediction
        # self.weights = nn.Parameter(
        #     torch.FloatTensor(self._input_dim, self._output_dim)
        # )
        # self.reset_parameters()
        # self.nodevec1 = nn.Parameter(torch.randn(self._num_nodes, 10))   #nodeEmbedding
        # self.nodevec2 = nn.Parameter(torch.randn(10, self._num_nodes))


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain("tanh"))

    def get_gloal_split_adj(self, globalAdj):
        clientNum = 7
        # startNodes = [0, 20, 50]  # 非平衡数据
        # endNodes = [20, 50, 156]
        eachNodeNum = 29  #shenzhen_52, 26, 17  losloop_69, 34, 23
        startNodes = []
        endNodes = []
        for i in range(clientNum):
            startNode = i * eachNodeNum
            endNode = (i + 1) * eachNodeNum
            startNodes.append(startNode)
            endNodes.append(endNode)

        splitGlobalAdj = []
        for i in range(clientNum):
            rowAdj = []
            rowLeftStart = startNodes[i]
            rowLeftEnd = endNodes[i]
            for j in range(clientNum):
                rowRightStart = startNodes[j]
                rowRightEnd = endNodes[j]
                currentAdj = globalAdj[rowLeftStart:rowLeftEnd, rowRightStart:rowRightEnd]
                rowAdj.append(currentAdj)
            splitGlobalAdj.append(rowAdj)


        # client0Start = 0
        # client0End = 52
        # client1Start = 52
        # client1End = 104
        # client2Start = 104
        # client2End = 156
        # # client0Start = 0
        # # client0End = 69
        # # client1Start = 69
        # # client1End = 138
        # # client2Start = 138
        # # client2End = 207
        # A00 = globalAdj[client0Start:client0End, client0Start:client0End]
        # A01 = globalAdj[client0Start:client0End, client1Start:client1End]
        # A02 = globalAdj[client0Start:client0End, client2Start:client2End]
        # B00 = globalAdj[client1Start:client1End, client0Start:client0End]
        # B01 = globalAdj[client1Start:client1End, client1Start:client1End]
        # B02 = globalAdj[client1Start:client1End, client2Start:client2End]
        # C00 = globalAdj[client2Start:client2End, client0Start:client0End]
        # C01 = globalAdj[client2Start:client2End, client1Start:client1End]
        # C02 = globalAdj[client2Start:client2End, client2Start:client2End]
        #
        # splitGlobalAdj = [[A00, A01, A02],
        #              [B00, B01, B02],
        #              [C00, C01, C02]]

        return splitGlobalAdj

    def getSpatialConv(self, splitGlobalAdj, inputs, id):
        globalSlice = []
        clientNum = 7

        for i in range(clientNum):
            currentSlice = splitGlobalAdj[i][id] @ inputs
            globalSlice += currentSlice
        globalSlice = torch.stack(globalSlice, 0)

        # if id == 0:
        #     AX_A00 = splitGlobalAdj[0][0] @ inputs
        #     AX_B00 = splitGlobalAdj[1][0] @ inputs
        #     AX_C00 = splitGlobalAdj[2][0] @ inputs
        #
        #     globalSlice += AX_A00
        #     globalSlice += AX_B00
        #     globalSlice += AX_C00
        #     globalSlice = torch.stack(globalSlice, 0)
        # elif id == 1:
        #     AX_A01 = splitGlobalAdj[0][1] @ inputs
        #     AX_B01 = splitGlobalAdj[1][1] @ inputs
        #     AX_C01 = splitGlobalAdj[2][1] @ inputs
        #
        #     globalSlice += AX_A01
        #     globalSlice += AX_B01
        #     globalSlice += AX_C01
        #     globalSlice = torch.stack(globalSlice, 0)
        # elif id == 2:
        #     AX_A02 = splitGlobalAdj[0][2] @ inputs
        #     AX_B02 = splitGlobalAdj[1][2] @ inputs
        #     AX_C02 = splitGlobalAdj[2][2] @ inputs
        #
        #     globalSlice += AX_A02
        #     globalSlice += AX_B02
        #     globalSlice += AX_C02
        #     globalSlice = torch.stack(globalSlice, 0)

        return globalSlice

    def forward(self, globalAdj, inputs, batch_size, output_dim, id):
        global_num_nodes = len(globalAdj)

        globalAdj = calculate_laplacian_with_self_loop(torch.FloatTensor(globalAdj))
        splitGlobalAdj = self.get_gloal_split_adj(globalAdj)
        outputs = self.getSpatialConv(splitGlobalAdj, inputs, id)
        # outputs = globalAdj @ inputs

        outputs = outputs.reshape((global_num_nodes, batch_size, output_dim))

        outputs = torch.tensor(outputs)
        outputs = outputs.transpose(0, 1)

        return outputs  # 返回空间卷积

# class GlobalGCN(nn.Module):
#     def __init__(self, adj, input_dim: int, output_dim: int, **kwargs):
#         super(GCN, self).__init__()
#
#         self.adj = adj
#         self.supports = 0
#         self._num_nodes = adj.shape[0]
#         self._input_dim = input_dim  # seq_len for prediction
#         self._output_dim = output_dim  # hidden_dim for prediction
#         self.weights = nn.Parameter(
#             torch.FloatTensor(self._input_dim, self._output_dim)
#         )
#         self.reset_parameters()
#         # self.nodevec1 = nn.Parameter(torch.randn(self._num_nodes, 10))   #nodeEmbedding
#         # self.nodevec2 = nn.Parameter(torch.randn(10, self._num_nodes))
#
#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain("tanh"))
#
#     def forward(self, globalAx, inputs, id):
#         ax = globalAx.reshape((self._num_nodes, batch_size, self._input_dim))
#         # (num_nodes * batch_size, seq_len)
#         ax = ax.reshape((self._num_nodes * batch_size, self._input_dim))
#         # act(AXW) (num_nodes * batch_size, output_dim)
#         outputs = torch.tanh(ax @ self.weights)
#         # (num_nodes, batch_size, output_dim)
#         outputs = outputs.reshape((self._num_nodes, batch_size, self._output_dim))
#         # (batch_size, num_nodes, output_dim)
#         outputs = outputs.transpose(0, 1)
#         return outputs

    # def forward(self, globalAdj, inputs, id):
    #     batch_size = inputs.shape[0]
    #
    #     adp = F.softmax(F.relu(torch.tensor(globalAdj)), dim=1)
    #     new_supports = adp
    #     global_num_nodes = len(new_supports)
    #
    #     inputs = inputs.reshape((self._num_nodes, batch_size * self._input_dim))
    #     splitGlobalAdj = self.get_gloal_split_adj(new_supports)
    #     outputs = self.getSpatialConv(splitGlobalAdj, inputs, id)
    #
    #     outputs = outputs.reshape((global_num_nodes, batch_size, self._input_dim))
    #
    #     outputs = torch.tensor(outputs)
    #     outputs = outputs.transpose(0, 1)
    #
    #     return outputs    #返回空间卷积


    # def forward(self, ax, inputs):
    #     # (batch_size, seq_len, num_nodes)
    #     batch_size = inputs.shape[0]
    #     # # (num_nodes, batch_size, seq_len)
    #     # inputs = inputs.transpose(0, 2).transpose(1, 2)
    #     # # (num_nodes, batch_size * seq_len)
    #     # inputs = inputs.reshape((self._num_nodes, batch_size * self._input_dim))
    #     # # AX (num_nodes, batch_size * seq_len)
    #     # ax = self.laplacian @ inputs
    #
    #     # (num_nodes, batch_size, seq_len)
    #     ax = ax.reshape((self._num_nodes, batch_size, self._input_dim))
    #     # (num_nodes * batch_size, seq_len)
    #     ax = ax.reshape((self._num_nodes * batch_size, self._input_dim))
    #     # act(AXW) (num_nodes * batch_size, output_dim)
    #     outputs = torch.tanh(ax @ self.weights)
    #     # (num_nodes, batch_size, output_dim)
    #     nodeEmbedding = outputs.reshape((self._num_nodes, batch_size, self._output_dim))
    #     # (batch_size, num_nodes, output_dim)
    #     outputs = nodeEmbedding.transpose(0, 1)
    #
    #     nodeEmbedding = nodeEmbedding.reshape((self._num_nodes, -1))
    #
    #     return outputs, nodeEmbedding


    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=100)
        return parser

    @property
    def hyperparameters(self):
        return {
            "num_nodes": self._num_nodes,
            "input_dim": self._input_dim,
            "output_dim": self._output_dim,
        }
