"""Experiment-running framework."""
from training.training import doTrain
import argparse

import numpy as np
import torch
#os.chdir('/home/seafood/wkdir/kg_trainer')
import sys
sys.path.append('.')
import wandb
import logging

from utils import HParam
from dataloader.data_loader import dataLoaderBase
from bertModel.loadBert import loadModel
import torch.nn as nn
loss_fn = nn.MultiLabelSoftMarginLoss()
#import nemo
#from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy

#from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy


# In order to ensure reproducible experiments, we must set random seeds.
np.random.seed(42)
torch.manual_seed(42)

def main():
    parser = argparse.ArgumentParser(add_help=False)


    parser.add_argument("--help", "-h", action="help")
    parser.add_argument("-c",'--config',default='config/default.yaml', type=str, help='set the config file')
    parser.add_argument("-m","--model_name", type=str, required= True, help='model name')

    args = parser.parse_args()
    hp = HParam(args.config)
    # 配置logging
    #wandb.init(project="InformationExtractor", config=hp)  # 替换为您的项目名和配置对象
    # 配置logging
    logging.basicConfig(level=logging.INFO)  # 设置全局日志级别为INFO
    logger = logging.getLogger(__name__)  # 获取当前模块的logger

    device = torch.device("cuda"if torch.cuda.is_available else"cpu")

    
    num_labels = hp.data.max_seq_length+8
    tokenizer,model = loadModel(hp.model,num_labels)
    #data
    train_loader,valid_loader = dataLoaderBase(tokenizer, hp.data)
    # Criterion
    doTrain(hp,model,train_loader,valid_loader,loss_fn,logger,device)
    # Call baks

if __name__ == "__main__":
    main()
