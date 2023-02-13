import argparse
import traceback
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
import models as models
import tasks as tasks
import utils as utils
import utils.callbacks
import utils.data
import utils.email
import utils.logging
import pandas as pd
import torch


def get_args(DATA_PATHS, startNode, endNode):
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument(
        "--data", type=str, help="The name of the dataset", choices=("shenzhen", "losloop"), default="losloop"
    )
    parser.add_argument(
        "--model_name_s",
        type=str,
        help="The name of the model for spatiotemporal prediction",
        choices=("GCN", "GRU", "TGCN"),
        default="GCN",
        # default="TGCN",
    )
    parser.add_argument(
        "--model_name_t",
        type=str,
        help="The name of the model for spatiotemporal prediction",
        choices=("GCN", "GRU", "TGCN"),
        default="GRU",
        # default="TGCN",
    )
    parser.add_argument(
        "--settings",
        type=str,
        help="The type of settings, e.g. supervised learning",
        choices=("supervised",),
        default="supervised",
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to the output console log file")
    parser.add_argument("--send_email", "--email", action="store_true", help="Send email when finished")

    temp_args, _ = parser.parse_known_args()

    parser = utils.data.SpatioTemporalCSVDataModule.add_data_specific_arguments(parser)
    parser = models.GCN.add_model_specific_arguments(parser)
    parser = tasks.SupervisedForecastTask.add_task_specific_arguments(parser)

    args = parser.parse_args()
    utils.logging.format_logger(pl._logger)
    if args.log_path is not None:
        utils.logging.output_logger_to_file(pl._logger, args.log_path)

    dm = utils.data.SpatioTemporalCSVDataModule(
        feat_path=DATA_PATHS[args.data]["feat"], adj_path=DATA_PATHS[args.data]["adj"], startNode=startNode,
        endNode=endNode,
        **vars(args)
    )
    return dm, args

class Client:
    def __init__(self, DATA_PATHS, startNode, endNode):
        self.DATA_PATHS = DATA_PATHS
        self.startNode = startNode
        self.endNode = endNode
        self.dm, args = get_args(self.DATA_PATHS, self.startNode, self.endNode)
        self.gcnModel, self.gruModel = self.get_model(args, self.dm)  #build GCN,GRU

    def get_model(self, args, dm):
        gcnModel = None
        gruModel = None
        finalGruModel = None
        if args.model_name_s == "GCN":
            gcnModel = models.GCN(adj=dm.adj, input_dim=args.seq_len, output_dim=args.hidden_dim)
        if args.model_name_t == "GRU":
            gruModel = models.GRU(input_dim=dm.adj.shape[0], hidden_dim=args.hidden_dim, feature_dim = 1)

        return gcnModel, gruModel


if __name__ == "__main__":
    DATA_PATHS = {
        "shenzhen": {"feat": "data/sz_speed.csv", "adj": "data/sz_adj.csv"},
        "losloop": {"feat": "data/los_speed.csv", "adj": "data/los_adj.csv"},
    }

    # client0 = Client(DATA_PATHS, 0, 69)
    # client1 = Client(DATA_PATHS, 69, 138)
    # client2 = Client(DATA_PATHS, 138, 207)
    # dm, args = get_args(DATA_PATHS, 0, 207)
    DS = 'losloop'
    clientNum = 7
    gruList = []
    gcnList = []
    client = []
    # startNodeNum = [0, 20, 50]   #非平衡数据
    # endNodeNum = [20, 50, 156]
    eachNodeNum = 29  #shenzhen_52_3， 26_6, 17_9    losloop_69_3, 34_6, 23_9
    for i in range(clientNum):
        startNode = i * eachNodeNum
        endNode = (i + 1) * eachNodeNum
        # startNode = startNodeNum[i]
        # endNode = endNodeNum[i]
        currentClient = Client(DATA_PATHS, startNode, endNode)
        client.append(currentClient)
        gruList.append(currentClient.gruModel)
        gcnList.append(currentClient.gcnModel)
    # client0 = Client(DATA_PATHS, 0, 52)
    # client1 = Client(DATA_PATHS, 52, 104)
    # client2 = Client(DATA_PATHS, 104, 156)
    # globalclient = Client(DATA_PATHS, 0, 156)
    # gruList = [client0.gruModel, client1.gruModel, client2.gruModel, globalclient.gruModel]
    # gcnList = [client0.gcnModel, client1.gcnModel, client2.gcnModel, globalclient.gcnModel]
    dm, args = get_args(DATA_PATHS, 0, 203) #shenzhen_9_153, losloop_6_204
    SFT = tasks.SupervisedForecastTask(gcnmodel=gcnList, grumodel=gruList,
                                       feat_max_val=dm.feat_max_val, **vars(args))
    optimizers = SFT.configure_optimizers()
    epoch = 0
    training_epoch = 0
    training_iteration = 100
    globalGlobalAdj = dm.adj
    totalbatch = int(len(dm.train_dataset) / dm.batch_size)

    for i in range(training_iteration):       #总的迭代次数
        for j in range(totalbatch):           #所有batch跑完所需要的次数
            mini_batch_list = []
            for k in range(clientNum):
                mini_batch = client[k].dm.train_dataset[j * dm.batch_size: (j + 1) * dm.batch_size]
                mini_batch_list.append(mini_batch)

            global_mini_batch = dm.train_dataset[j * dm.batch_size: (j + 1) * dm.batch_size]
            loss, globalAdj = SFT.training_step(mini_batch_list, global_mini_batch, epoch, globalGlobalAdj)
            loss.backward()
            optimizers.step()
            optimizers.zero_grad()

        #一次整体迭代结束后，就开始验证
        valLen = len(dm.val_dataset)
        mini_batch = dm.val_dataset[0:valLen]
        valList = []
        for k in range(clientNum):
            current_mini_batch = client[k].dm.val_dataset[0:valLen]
            valList.append(current_mini_batch)
        loss = SFT.validation_step(valList, globalGlobalAdj, mini_batch, i)


    dataframe = pd.DataFrame(
                    {'loss': SFT.val_loss, 'acc': SFT.accuracy,
                     'rmse': SFT.rmse, 'mae': SFT.mae, 'r2': SFT.r2})  # save data

    # dataframe.to_csv('multiClientResult\client9_LOS_SST(ADP+FedAvg)_Result.csv', index=False, sep=',')
    dataframe.to_csv('multiClientResult\client7_LOS_SST(ADP_FedAvg)_Result.csv', index=False, sep=',')




















