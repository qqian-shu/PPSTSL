import argparse
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import utils as utils
import utils.metrics
import utils.losses
import utils as utils
import utils.data
import models as models
from utils.graph_conv import calculate_laplacian_with_self_loop
import numpy as np
import pandas as pd

class SupervisedForecastTask(pl.LightningModule):
    def __init__(
        self,
        gcnmodel: [nn.ModuleList],
        grumodel: [nn.ModuleList],
        regressor="linear",
        loss="mse",
        pre_len: int = 3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1.5e-3,
        feat_max_val: float = 1.0,
        **kwargs
    ):
        super(SupervisedForecastTask, self).__init__()
        self.save_hyperparameters()
        self.hidden_dim = 100
        self.input_dim = 100
        self.global_feature_dim = 100
        self.global_input_dim = 203
        self.seq_len = 12
        self.pre_len = 3
        self.num_nodes = 203
        self.gcnmodel = gcnmodel
        self.grumodel = grumodel
        self.grumodel0 = grumodel[0]
        self.grumodel1 = grumodel[1]
        self.grumodel2 = grumodel[2]
        self.grumodel3 = grumodel[0]
        self.grumodel4 = grumodel[4]
        self.grumodel5 = grumodel[5]
        self.grumodel6 = grumodel[6]
        # self.grumodel7 = grumodel[7]
        # self.grumodel8 = grumodel[8]
        self.globalGruModel = models.GRU(input_dim=self.global_input_dim, hidden_dim=self.hidden_dim, feature_dim = self.global_feature_dim)
        self.globalGruModelSecond = models.GRU(input_dim=self.global_input_dim, hidden_dim=self.hidden_dim,
                                         feature_dim=self.global_feature_dim)
        self.regressor = (
            nn.Linear(
                self.hidden_dim,
                self.hparams.pre_len,
            )
            if regressor == "linear"
            else regressor
        )

        #######weight#######
        self.weights = nn.Parameter(torch.FloatTensor(self.seq_len, self.pre_len))
        self.biases = nn.Parameter(torch.FloatTensor(self.pre_len))
        ########att########
        self.weights_att0 = nn.Parameter(
            torch.FloatTensor(self.hidden_dim, 1)
        )
        self.biases_att0 = nn.Parameter(torch.FloatTensor(1))

        self.weights_att1 = nn.Parameter(
            torch.FloatTensor(self.global_input_dim, 1)
        )
        self.biases_att1 = nn.Parameter(torch.FloatTensor(1))

        nn.init.xavier_uniform_(self.weights_att0)
        nn.init.xavier_uniform_(self.weights_att1)
        nn.init.xavier_uniform_(self.weights)
        self._bias_init_value = 0
        nn.init.constant_(self.biases_att0, self._bias_init_value)
        nn.init.constant_(self.biases_att1, self._bias_init_value)
        nn.init.constant_(self.biases, self._bias_init_value)

        self._loss = loss
        self.feat_max_val = feat_max_val
        self.val_loss = []
        self.rmse = []
        self.mae = []
        self.accuracy = []
        self.r2 = []
        self.predictionList = []
        self.yRealList = []
        self.accuracy_record = 0

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights_att)
        nn.init.constant_(self.biases_att, 0.0)

    def getEudieanDistance(self, samples):
        samplesNum = len(samples)
        distance = []
        for i in range(samplesNum):
            rowDis = []
            for j in range(samplesNum):
                sampleX = samples[i]
                sampleY = samples[j]
                dis = np.sum(np.square(sampleX - sampleY))
                rowDis.append(dis)
            distance.append(rowDis)
        distance = np.array(distance)

        return distance

    def get_innerProduct(self, emb):
        threshold = 0.01 #0.0001 #4.6   #0.5(3)   #0.05(9)  #0.001(9)
        emb = np.array(emb)
        adj_rec = np.matmul(emb, emb.T)  # innerProduct
        # adj_rec = self.getEudieanDistance(emb)

        adj_rec[adj_rec <= threshold] = 0
        adj_rec[adj_rec > threshold] = 1

        return adj_rec

    def getGlobalAdj(self, nodeEmbeddingList):
        currentNodeEmbedding = np.vstack((nodeEmbeddingList[0], nodeEmbeddingList[1]))
        clientNum = len(nodeEmbeddingList)
        for i in range(2, clientNum):
            currentNodeEmbedding = np.vstack((currentNodeEmbedding, nodeEmbeddingList[2]))

        currentGlobalAdj = self.get_innerProduct(currentNodeEmbedding)

        return currentGlobalAdj

    def setFedGlobalPar(self):
        globalWeights1 = 0
        globalBiases1 = 0
        globalWeights2 = 0
        globalBiases2 = 0
        n = len(self.grumodel)
        # grumodelList = [self.grumodel0, self.grumodel1, self.grumodel2, self.grumodel3, self.grumodel4, self.grumodel5, self.grumodel6, self.grumodel7, self.grumodel8]
        # grumodelList = [self.grumodel0, self.grumodel1, self.grumodel2]
        # grumodelList = [self.grumodel0, self.grumodel1, self.grumodel2, self.grumodel3, self.grumodel4]
        grumodelList = [self.grumodel0, self.grumodel1, self.grumodel2, self.grumodel3, self.grumodel4, self.grumodel5, self.grumodel6]
        for i in range(n):
            currentModelWeights1 = grumodelList[i].gru_cell.linear1.weights
            currentBiases1 = grumodelList[i].gru_cell.linear1.biases
            currentModelWeights2 = grumodelList[i].gru_cell.linear2.weights
            currentBiases2 = grumodelList[i].gru_cell.linear2.biases

            currentModelWeights1 = currentModelWeights1.detach().numpy()
            currentBiases1 = currentBiases1.detach().numpy()
            currentModelWeights2 = currentModelWeights2.detach().numpy()
            currentBiases2 = currentBiases2.detach().numpy()

            globalWeights1 += currentModelWeights1
            currentBiases1 += currentBiases1
            globalWeights2 += currentModelWeights2
            currentBiases2 += currentBiases2
        globalWeights1 = globalWeights1 / n
        globalBiases1 = globalBiases1 / n
        globalWeights2 = globalWeights2 / n
        globalBiases2 = globalBiases2 / n

        self.grumodel0.gru_cell.linear1.weights.data = torch.from_numpy(np.array(globalWeights1)).to(
            torch.float32)
        self.grumodel0.gru_cell.linear1.biases.data = torch.from_numpy(np.array(globalBiases1)).to(
            torch.float32)
        self.grumodel0.gru_cell.linear2.weights.data = torch.from_numpy(np.array(globalWeights2)).to(
            torch.float32)
        self.grumodel0.gru_cell.linear2.biases.data = torch.from_numpy(np.array(globalBiases2)).to(
            torch.float32)

        self.grumodel1.gru_cell.linear1.weights.data = torch.from_numpy(np.array(globalWeights1)).to(
            torch.float32)
        self.grumodel1.gru_cell.linear1.biases.data = torch.from_numpy(np.array(globalBiases1)).to(
            torch.float32)
        self.grumodel1.gru_cell.linear2.weights.data = torch.from_numpy(np.array(globalWeights2)).to(
            torch.float32)
        self.grumodel1.gru_cell.linear2.biases.data = torch.from_numpy(np.array(globalBiases2)).to(
            torch.float32)

        self.grumodel2.gru_cell.linear1.weights.data = torch.from_numpy(np.array(globalWeights1)).to(
            torch.float32)
        self.grumodel2.gru_cell.linear1.biases.data = torch.from_numpy(np.array(globalBiases1)).to(
            torch.float32)
        self.grumodel2.gru_cell.linear2.weights.data = torch.from_numpy(np.array(globalWeights2)).to(
            torch.float32)
        self.grumodel2.gru_cell.linear2.biases.data = torch.from_numpy(np.array(globalBiases2)).to(
            torch.float32)

        self.grumodel3.gru_cell.linear1.weights.data = torch.from_numpy(np.array(globalWeights1)).to(
            torch.float32)
        self.grumodel3.gru_cell.linear1.biases.data = torch.from_numpy(np.array(globalBiases1)).to(
            torch.float32)
        self.grumodel3.gru_cell.linear2.weights.data = torch.from_numpy(np.array(globalWeights2)).to(
            torch.float32)
        self.grumodel3.gru_cell.linear2.biases.data = torch.from_numpy(np.array(globalBiases2)).to(
            torch.float32)

        self.grumodel4.gru_cell.linear1.weights.data = torch.from_numpy(np.array(globalWeights1)).to(
            torch.float32)
        self.grumodel4.gru_cell.linear1.biases.data = torch.from_numpy(np.array(globalBiases1)).to(
            torch.float32)
        self.grumodel4.gru_cell.linear2.weights.data = torch.from_numpy(np.array(globalWeights2)).to(
            torch.float32)
        self.grumodel4.gru_cell.linear2.biases.data = torch.from_numpy(np.array(globalBiases2)).to(
            torch.float32)

        self.grumodel5.gru_cell.linear1.weights.data = torch.from_numpy(np.array(globalWeights1)).to(
            torch.float32)
        self.grumodel5.gru_cell.linear1.biases.data = torch.from_numpy(np.array(globalBiases1)).to(
            torch.float32)
        self.grumodel5.gru_cell.linear2.weights.data = torch.from_numpy(np.array(globalWeights2)).to(
            torch.float32)
        self.grumodel5.gru_cell.linear2.biases.data = torch.from_numpy(np.array(globalBiases2)).to(
            torch.float32)

        self.grumodel6.gru_cell.linear1.weights.data = torch.from_numpy(np.array(globalWeights1)).to(
            torch.float32)
        self.grumodel6.gru_cell.linear1.biases.data = torch.from_numpy(np.array(globalBiases1)).to(
            torch.float32)
        self.grumodel6.gru_cell.linear2.weights.data = torch.from_numpy(np.array(globalWeights2)).to(
            torch.float32)
        self.grumodel6.gru_cell.linear2.biases.data = torch.from_numpy(np.array(globalBiases2)).to(
            torch.float32)

        # self.grumodel7.gru_cell.linear1.weights.data = torch.from_numpy(np.array(globalWeights1)).to(
        #     torch.float32)
        # self.grumodel7.gru_cell.linear1.biases.data = torch.from_numpy(np.array(globalBiases1)).to(
        #     torch.float32)
        # self.grumodel7.gru_cell.linear2.weights.data = torch.from_numpy(np.array(globalWeights2)).to(
        #     torch.float32)
        # self.grumodel7.gru_cell.linear2.biases.data = torch.from_numpy(np.array(globalBiases2)).to(
        #     torch.float32)

        # self.grumodel8.gru_cell.linear1.weights.data = torch.from_numpy(np.array(globalWeights1)).to(
        #     torch.float32)
        # self.grumodel8.gru_cell.linear1.biases.data = torch.from_numpy(np.array(globalBiases1)).to(
        #     torch.float32)
        # self.grumodel8.gru_cell.linear2.weights.data = torch.from_numpy(np.array(globalWeights2)).to(
        #     torch.float32)
        # self.grumodel8.gru_cell.linear2.biases.data = torch.from_numpy(np.array(globalBiases2)).to(
        #     torch.float32)


    def getIFAtoGraph(self, gruFeature, initAdj):
        adjSum = np.sum(initAdj, axis=0)
        # sortedIndex = sorted(range(len(adjSum)), key=lambda x: adjSum[x], reverse=True)
        nodeNums = len(adjSum)
        featureList = []
        for i in range(nodeNums):
            row = initAdj[i]
            adjacentIndex = np.nonzero(row)
            currentIndex = i
            currentNodeSum = adjSum[i]
            for j in range(len(adjacentIndex)):
                currentAdjacentIndex = adjacentIndex[0][j]
                adjacentSum = adjSum[currentAdjacentIndex]
                if currentNodeSum < adjacentSum:
                    currentNodeSum = adjacentSum
                    currentIndex = currentAdjacentIndex
            featureList.append(gruFeature[currentIndex])
        featureList = torch.stack(featureList, 0)
        return featureList

    def self_attention1(self, x):
        seq_len = x.shape[1]
        x = x.reshape(-1, self.hidden_dim)
        x = x @ self.weights_att0 + self.biases_att0
        fx = x.reshape(-1, self.global_input_dim)
        f = fx @ self.weights_att1 + self.biases_att1
        g = f

        f1 = f.reshape(-1, seq_len)
        g1 = g.reshape(-1, seq_len)

        s = g1 * f1
        beta = nn.Softmax(dim=1)(s)
        xxx0 = torch.unsqueeze(beta, 2)
        xxx1 = x.reshape(-1, seq_len, self.global_input_dim)

        context = xxx0 * xxx1
        context = context.transpose(2, 1)

        return context, beta

    def forward(self, xList, globalX, initAdj):
        clientX = []
        clietnY = []
        clientNum = len(xList)
        for i in range(clientNum):
            currentClientX, currentClientY = xList[i]
            clientX.append(currentClientX)
            clietnY.append(currentClientY)
        self.setFedGlobalPar()

        localGruOutput = []
        # for i in range(clientNum):   #get local gru output
        #     outputs, currentClientTd = self.grumodel[i](clientX[i])
        #     localGruOutput.append(outputs)
        outputs, currentClientTd = self.grumodel0(clientX[0])
        localGruOutput.append(outputs)
        outputs, currentClientTd = self.grumodel1(clientX[1])
        localGruOutput.append(outputs)
        outputs, currentClientTd = self.grumodel2(clientX[2])
        localGruOutput.append(outputs)
        outputs, currentClientTd = self.grumodel3(clientX[3])
        localGruOutput.append(outputs)
        outputs, currentClientTd = self.grumodel4(clientX[4])
        localGruOutput.append(outputs)
        outputs, currentClientTd = self.grumodel5(clientX[5])
        localGruOutput.append(outputs)
        outputs, currentClientTd = self.grumodel6(clientX[6])
        localGruOutput.append(outputs)
        # outputs, currentClientTd = self.grumodel7(clientX[7])
        # localGruOutput.append(outputs)
        # outputs, currentClientTd = self.grumodel8(clientX[8])
        # localGruOutput.append(outputs)

        #get new input gru feature
        seq_len = clientX[0].shape[1]
        newInput = []
        seq_adj = []
        seq_residual = []
        for i in range(seq_len):
            residual = []
            nodeEmbeddingList = []
            clientTdFAdj = []
            for j in range(clientNum):
                currentClientTd = localGruOutput[j][i]
                currentClientTd = currentClientTd.transpose(0, 1)
                residual += currentClientTd
                currentClientTdFAdj = currentClientTd.reshape(len(currentClientTd), -1)
                clientTdFAdj.append(currentClientTdFAdj)
                nodeEmbeddingList.append(currentClientTdFAdj.detach().numpy())
            globalSeqAdj = self.getGlobalAdj(nodeEmbeddingList)   #getting the current global nodes relationship
            # globalSeqAdj = initAdj

            seq_adj.append(globalSeqAdj)
            residual = torch.stack(residual, 0)  #156,32,100
            # residual = self.getIFAtoGraph(residual, initAdj)
            seq_residual.append(residual)
            batch_size = residual.shape[1]
            output_dim = residual.shape[2]
            nodeNum = residual.shape[0]

            globalAx = 0
            for j in range(clientNum):   #getting global convolution
                client0Sd = self.gcnmodel[j](globalSeqAdj, clientTdFAdj[j], batch_size, output_dim, j)
                globalAx += client0Sd

            globalAx = globalAx.transpose(0, 1)   #32,156,100

            currentNewInput = globalAx + residual
            # currentNewInput = torch.stack([residual, globalAx], dim=2)
            # currentNewInput = currentNewInput.reshape(nodeNum, batch_size, -1)

            currentNewInput = currentNewInput.transpose(1, 0)
            newInput.append(currentNewInput)
        newInput = torch.stack(newInput, 0)
        newInput = newInput.transpose(1, 0)
        globalAdj = initAdj

        globalGruOutput, lastGlobalGruOutput = self.globalGruModel(newInput)

        newInput11 = torch.stack(globalGruOutput, 0)
        seq_residual = torch.stack(seq_residual, 0)
        seq_residual = seq_residual.transpose(2, 1)

        newInput11 = newInput11 + seq_residual
        # newInput11 = torch.stack([seq_residual, newInput11], dim=3)
        # newInput11 = newInput11.reshape(seq_len, batch_size, nodeNum, -1)

        newInput11 = newInput11.transpose(1, 0)
        globalGruOutput, lastGlobalGruOutput = self.globalGruModelSecond(newInput11)

        for i in range(1):
            newInput11 = torch.stack(globalGruOutput, 0)
            newInput11 = newInput11 + seq_residual
            # newInput11 = torch.stack([seq_residual, newInput11], dim=3)
            # newInput11 = newInput11.reshape(seq_len, batch_size, nodeNum, -1)

            newInput11 = newInput11.transpose(1, 0)
            globalGruOutput, lastGlobalGruOutput = self.globalGruModelSecond(newInput11)

        ##########加att##########################
        # out = torch.stack(globalGruOutput, 0)
        # out = out.transpose(1, 0)
        # lastGlobalGruOutput, alpha = self.self_attention1(out)
        # lastGlobalGruOutput = lastGlobalGruOutput.reshape(-1, seq_len)
        # output = lastGlobalGruOutput @ self.weights + self.biases
        # predictions = output.reshape(-1, self.num_nodes, self.pre_len)

        ######没加att的时候####################
        num_nodes = lastGlobalGruOutput.size(1)
        hidden = lastGlobalGruOutput.reshape((-1, lastGlobalGruOutput.size(2)))

        if self.regressor is not None:
            predictions = self.regressor(hidden)
        else:
            predictions = hidden
        predictions = predictions.reshape((batch_size, num_nodes, -1))
        return predictions, globalAdj

    def shared_step(self, batch_list, globalX, batch_idx, initAdj):
        x, y = globalX
        num_nodes = x.size(2)
        predictions, globalAdj = self(batch_list, globalX, initAdj)
        predictions = predictions.transpose(1, 2).reshape((-1, num_nodes))
        y = y.reshape((-1, y.size(2)))
        return predictions, y, globalAdj

    def loss(self, inputs, targets):
        if self._loss == "mse":
            return F.mse_loss(inputs, targets)
        if self._loss == "mse_with_regularizer":
            return utils.losses.mse_with_regularizer_loss(inputs, targets, self)
        raise NameError("Loss not supported:", self._loss)

    def getGlobalAx(self, globalAdj, batch, id):
        x, y = batch
        globalAxSlice = self.model(globalAdj, x, id)
        return globalAxSlice

    def training_step(self, batch_list, globalX, batch_idx, initAdj):
        predictions, y, globalAdj = self.shared_step(batch_list, globalX, batch_idx, initAdj)
        loss = self.loss(predictions, y)
        self.log("train_loss", loss)
        return loss, globalAdj

    def validate(self, valList, globalAdj, globalValX):
        x, y = globalValX
        clientNum = len(valList)
        clientX = []
        clientY = []
        for i in range(clientNum):
            currentClientX, currentClientY = valList[i]
            clientX.append(currentClientX)
            clientY.append(currentClientY)

        localGruOutput = []
        outputs, currentClientTd = self.grumodel0(clientX[0])
        localGruOutput.append(outputs)
        outputs, currentClientTd = self.grumodel1(clientX[1])
        localGruOutput.append(outputs)
        outputs, currentClientTd = self.grumodel2(clientX[2])
        localGruOutput.append(outputs)
        outputs, currentClientTd = self.grumodel3(clientX[3])
        localGruOutput.append(outputs)
        outputs, currentClientTd = self.grumodel4(clientX[4])
        localGruOutput.append(outputs)
        outputs, currentClientTd = self.grumodel5(clientX[5])
        localGruOutput.append(outputs)
        outputs, currentClientTd = self.grumodel6(clientX[6])
        localGruOutput.append(outputs)
        # outputs, currentClientTd = self.grumodel7(clientX[7])
        # localGruOutput.append(outputs)
        # outputs, currentClientTd = self.grumodel8(clientX[8])
        # localGruOutput.append(outputs)

        # get new input gru feature
        seq_len = clientX[0].shape[1]
        newInput = []
        seq_adj = []
        seq_residual = []
        for i in range(seq_len):
            residual = []
            nodeEmbeddingList = []
            clientTdFAdj = []
            for j in range(clientNum):
                currentClientTd = localGruOutput[j][i]
                currentClientTd = currentClientTd.transpose(0, 1)
                residual += currentClientTd
                currentClientTdFAdj = currentClientTd.reshape(len(currentClientTd), -1)
                clientTdFAdj.append(currentClientTdFAdj)
                nodeEmbeddingList.append(currentClientTdFAdj.detach().numpy())
            globalSeqAdj = self.getGlobalAdj(nodeEmbeddingList)  # 得到当前序列的全局邻接关系
            # globalSeqAdj = globalAdj
            seq_adj.append(globalSeqAdj)
            residual = torch.stack(residual, 0)  # 156,32,100
            # residual = self.getIFAtoGraph(residual, globalAdj)
            seq_residual.append(residual)
            batch_size = residual.shape[1]
            output_dim = residual.shape[2]
            nodeNum = residual.shape[0]

            globalAx = 0
            for j in range(clientNum):  # getting global convolution
                client0Sd = self.gcnmodel[j](globalSeqAdj, clientTdFAdj[j], batch_size, output_dim, j)
                globalAx += client0Sd

            globalAx = globalAx.transpose(0, 1)  # 32,156,100
            # currentNewInput = torch.stack([residual, globalAx], dim=2)
            # currentNewInput = currentNewInput.reshape(nodeNum, batch_size, -1)
            currentNewInput = globalAx + residual
            currentNewInput = currentNewInput.transpose(1, 0)
            newInput.append(currentNewInput)
        newInput = torch.stack(newInput, 0)
        newInput = newInput.transpose(1, 0)
        globalAdj = globalAdj

        globalGruOutput, lastGlobalGruOutput = self.globalGruModel(newInput)
                # newInput11 = torch.stack(globalGruOutput, 0)
                # newInput11 = newInput11.transpose(1, 0)
        newInput11 = torch.stack(globalGruOutput, 0)
        seq_residual = torch.stack(seq_residual, 0)
        seq_residual = seq_residual.transpose(2, 1)
        newInput11 = newInput11 + seq_residual
        # newInput11 = torch.stack([seq_residual, newInput11], dim=3)
        # newInput11 = newInput11.reshape(seq_len, batch_size, nodeNum, -1)

        newInput11 = newInput11.transpose(1, 0)
        globalGruOutput, lastGlobalGruOutput = self.globalGruModelSecond(newInput11)

        for i in range(1):
            newInput11 = torch.stack(globalGruOutput, 0)
            newInput11 = newInput11 + seq_residual
            # newInput11 = torch.stack([seq_residual, newInput11], dim=3)
            # newInput11 = newInput11.reshape(seq_len, batch_size, nodeNum, -1)

            newInput11 = newInput11.transpose(1, 0)
            globalGruOutput, lastGlobalGruOutput = self.globalGruModelSecond(newInput11)

        num_nodes = lastGlobalGruOutput.size(1)
        hidden = lastGlobalGruOutput.reshape((-1, lastGlobalGruOutput.size(2)))

        if self.regressor is not None:
            predictions = self.regressor(hidden)
        else:
            predictions = hidden
        predictions = predictions.reshape((batch_size, num_nodes, -1))
        predictions = predictions.transpose(1, 2).reshape((-1, num_nodes))
        y = y.reshape((-1, y.size(2)))
        return predictions, y

    def validation_step(self, valList, globalAdj, globalValX, batch_idx):
        print('val_index:', batch_idx)
        predictions, y = self.validate(valList, globalAdj, globalValX)

        predictions = predictions * self.feat_max_val
        y = y * self.feat_max_val

        loss = self.loss(predictions, y)
        rmse = torch.sqrt(torchmetrics.functional.mean_squared_error(predictions, y))
        mae = torchmetrics.functional.mean_absolute_error(predictions, y)
        accuracy = utils.metrics.accuracy(predictions, y)
        r2 = utils.metrics.r2(predictions, y)
        explained_variance = utils.metrics.explained_variance(predictions, y)

        if accuracy.item() > self.accuracy_record:
            self.accuracy_record = accuracy.item()

            currentPredic = predictions.detach().numpy()
            currentY = y.detach().numpy()

            dataframe0 = pd.DataFrame(currentPredic)  # save data
            # dataframe0.to_csv('unFixedSplitting\client3_SZ_SST(PD_FedAvg)_Prediction_Result.csv', index=False, sep=',')
            dataframe0.to_csv('multiClientResult\client7_LOS_SST(ADP_FedAvg)_Prediction_Result.csv', index=False, sep=',')
            dataframe1 = pd.DataFrame(currentY)  # save data
            # dataframe1.to_csv('unFixedSplitting\client3_SZ_SST(PD_FedAvg)_Y_Result.csv', index=False, sep=',')
            dataframe1.to_csv('multiClientResult\client7_LOS_SST(ADP_FedAvg)_Y_Result.csv', index=False, sep=',')
        metrics = {
            "val_loss": loss,
            "RMSE": rmse,
            "MAE": mae,
            "accuracy": accuracy,
            "R2": r2,
            "ExplainedVar": explained_variance,
        }

        self.val_loss.append(loss.item())
        self.rmse.append(rmse.item())
        self.mae.append(mae.item())
        self.accuracy.append(accuracy.item())
        self.r2.append(r2.item())

        print('val_loss:', loss.item(), 'RMSE:', rmse.item(), 'MAE:',mae.item(), 'accuracy:',accuracy.item(), 'R2:',r2.item())
        self.log_dict(metrics)
        # return predictions.reshape(batch[1].size()), y.reshape(batch[1].size())

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    @staticmethod
    def add_task_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", "--lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", "--wd", type=float, default=1.5e-3)
        parser.add_argument("--loss", type=str, default="mse")
        return parser
