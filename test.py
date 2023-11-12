"""Experiment-running framework."""
from training.training import doTrain
from training.testing import doTest
import argparse

import numpy as np
import torch
#os.chdir('/home/seafood/wkdir/kg_trainer')
import sys
sys.path.append('.')
import wandb
import logging
import importlib
import json

from utils import HParam
import torch.nn as nn
#loss_fn = nn.MultiLabelSoftMarginLoss()
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
#import nemo
# In order to ensure reproducible experiments, we must set random seeds.
np.random.seed(42)
torch.manual_seed(42)
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def main():
    parser = argparse.ArgumentParser(add_help=False)


    parser.add_argument("--help", "-h", action="help")
    parser.add_argument("-c",'--config',default='config/default.yaml', type=str, help='set the config file')
    parser.add_argument("-m","--model_name", type=str, required= True, help='model name')
    parser.add_argument("--checkpoint", type=str, required= True, help='checkpoint name')

    args = parser.parse_args()
    hp = HParam(args.config)
    # 配置logging
    #wandb.init(project="InformationExtractor", config=hp)  # 替换为您的项目名和配置对象
    # 配置logging
    logging.basicConfig(level=logging.INFO)  # 设置全局日志级别为INFO
    logger = logging.getLogger(__name__)  # 获取当前模块的logger

    device = torch.device("cuda"if torch.cuda.is_available else"cpu")
    device = torch.device("mps") if torch.backends.mps.is_available else device
    print("using device as ",device)

    #os.mkdir(f'outputs/model/{args.model_name}')
    hp.data.num_labels = hp.model.num_classes
    hp.data.debug_mode = hp.trainer.debug_mode
    hp.model.max_seq_length = hp.data.max_seq_length
    with open("./data/duie/predicate2id.json", 'r', encoding='utf8') as fp:
        label_map = json.load(fp)
    
    hp.model.num_relation = len(label_map.keys()) 
    print('关系的数量为',hp.model.num_relation)

    loadModel = getattr(importlib.import_module(hp.model.lib_name),'loadModel')
    dataLoaderBase = getattr(importlib.import_module(hp.data.lib_data),'dataLoaderBase')
    tokenizer,model = loadModel(hp.model)

    model = torch.load(args.checkpoint,map_location=device)
    test_loader = dataLoaderBase(tokenizer, hp.data,'do_test')
    doTest(hp,model,test_loader,loss_fn,logger,device)

if __name__ == "__main__":
    main()
