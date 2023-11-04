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
    parser.add_argument("--do_test", type=bool, default=False, help='whether do the test step only')

    parser.add_argument("--do_train", type=bool, default=True, help='whether do the test step only')

    args = parser.parse_args()
    hp = HParam(args.config)
    # 配置logging
    #wandb.init(project="InformationExtractor", config=hp)  # 替换为您的项目名和配置对象
    # 配置logging
    logging.basicConfig(level=logging.INFO)  # 设置全局日志级别为INFO
    logger = logging.getLogger(__name__)  # 获取当前模块的logger

    device = torch.device("mps"if torch.backends.mps.is_available else"cpu")

    #os.mkdir(f'outputs/model/{args.model_name}')
    hp.data.num_labels = hp.model.num_classes
    hp.data.debug_mode = hp.trainer.debug_mode
    hp.model.max_seq_length = hp.data.max_seq_length
    with open("./data/duie/predicate2id.json", 'r', encoding='utf8') as fp:
        label_map = json.load(fp)
    
    hp.model.num_relation = 2 * (len(label_map.keys()) - 2) + 2
    print('关系的数量为{}'%{hp.model.num_relation})

    loadModel = getattr(importlib.import_module(hp.model.lib_name),'loadModel')
    dataLoaderBase = getattr(importlib.import_module(hp.data.lib_data),'dataLoaderBase')
    tokenizer,model = loadModel(hp.model)

    if args.do_test:
        model = torch.load(hp.model.ckpt_dir)
        test_loader = dataLoaderBase(tokenizer, hp.data,'do_test')
        doTest(hp,model,test_loader,loss_fn,logger,device)
        return 0
    #data
    train_loader,valid_loader = dataLoaderBase(tokenizer, hp.data,'do_train')
    # Criterion
    #wandb.init()
    if args.do_train:
        doTrain(hp,model,train_loader,valid_loader,loss_fn,logger,device,name=args.model_name)
    
    # Call baks

if __name__ == "__main__":
    main()
