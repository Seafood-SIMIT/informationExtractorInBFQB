"""Experiment-running framework."""
import argparse
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
import os
#os.chdir('/home/seafood/wkdir/kg_trainer')
from pytorch_lightning.callbacks import LearningRateMonitor
import sys
sys.path.append('.')

from utils import HParam
from dataloader.data_loader_bak import BaseDataModule
from bertModel.loadBert import loadModel
from modules.bertModule import BertModule
#import nemo
#from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy

#from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy


# In order to ensure reproducible experiments, we must set random seeds.
np.random.seed(42)
torch.manual_seed(42)


@rank_zero_only
def _ensure_logging_dir(experiment_dir):
    """Create the logging directory via the rank-zero process, if necessary."""
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(add_help=False)


    parser.add_argument("--help", "-h", action="help")
    parser.add_argument("-c",'--config',default='config/default.yaml', type=str, help='set the config file')
    parser.add_argument("-m","--model_name", type=str, required= True, help='model name')

    args = parser.parse_args()
    hp = HParam(args.config)

    num_labels = hp.data.max_seq_length+8
    tokenizer,model = loadModel(hp.model,num_labels)
    #data
    data = BaseDataModule(tokenizer, hp.data)
    #data = WenzhongQADataModel(hp.data, tokenizer)
    gpt2_litmodel = BertModule

    print('load from hugging face')
    gpt2_litmodel = gpt2_litmodel(args=hp.model, model=model,num_data=4)

    # Call baks
    log_dir = 'logs'
    _ensure_logging_dir(log_dir)
    logger = pl.loggers.WandbLogger(project='InformationExtraction',name=args.model_name,save_dir=log_dir)

    #goldstar_metric = "validation/cer" if hp.gpt2.loss in ("transformer",) else "val_loss"
    filename_format = "{epoch:04d}-{val_loss:.3f}"
    hp.ckpt.file_name = filename_format
    arg = hp.ckpt
    hp.ckpt.dirpath = hp.ckpt.dirpath+'/'+args.model_name
    checkpoint_callback = ModelCheckpoint(monitor=arg.monitor,
                                         save_top_k=arg.save_top_k,
                                         mode=arg.mode,
                                         every_n_train_steps=arg.every_n_train_steps,
                                         save_weights_only=arg.save_weights_only,
                                         dirpath=arg.dirpath,
                                         filename=arg.file_name,
                                         save_last=arg.save_last)

    summary_callback = pl.callbacks.ModelSummary(max_depth=2)


    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [summary_callback, checkpoint_callback, lr_monitor]
    if hp.trainer.stop_early:
        early_stopping_callback = pl.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=args.stop_early
        )
        callbacks.append(early_stopping_callback)

    if hp.trainer.debug_mode:
        callbacks = []

    trainer = pl.Trainer(#devices=hp.trainer.devices,
                        accelerator=hp.trainer.accelerator,
                        max_epochs=hp.trainer.max_epochs,
                        #strategy=hp.trainer.strategy,
                        precision=hp.trainer.fp,
                        #accumulate_grad_batches=hp.trainer.gas,
                        #strategy=NLPDDPStrategy(),
                        callbacks=callbacks,
                        logger = logger)

    #trainer.tune(lit_model, datamodule=data)  # If passing --auto_lr_find, this will set learning rate
    

    trainer.fit(gpt2_litmodel, datamodule=data)

    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        rank_zero_info(f"Best model saved at: {best_model_path}")
        trainer.test(datamodule=data, ckpt_path=best_model_path)
    else:
        trainer.test(gpt2_litmodel, datamodule=data)


if __name__ == "__main__":
    main()
