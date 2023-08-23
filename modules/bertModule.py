import torch
import pytorch_lightning as pl
from transformers.optimization import get_linear_schedule_with_warmup
import torch.nn as nn
loss_fn = nn.MultiLabelSoftMarginLoss()
class BertModule(pl.LightningModule):

    def __init__(self, args,model, num_data):
        super().__init__()
        self.args = args
        self.num_data = num_data
        print('num_data:', num_data)
        self.model = model
        #self.save_hyperparameters()
        

    def setup(self, stage) -> None:
        if stage == 'fit':
            #num_gpus = self.trainer.gpus if self.trainer.gpus is not None else 0
            self.total_step = int(self.trainer.max_epochs * self.num_data
                                  / (max(1, 1) * self.trainer.accumulate_grad_batches))
            print('Total training step:', self.total_step)

    def forward(self,x):
        output = self.model(**x)
        return output.logits
        
    def training_step(self, batch, batch_idx):
        output = self.model(
            input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        # output = self.model(input_ids=batch['input_ids'], labels=batch['labels'])
        # acc = self.comput_metrix(output.logits, batch['labels'])
        logits = output.logits
        
        loss = self.countLoss(logits,batch['labels'].float())
        self.log('train_loss', loss,on_epoch=True, prog_bar=True,logger=True)
        return output.loss

    def countLoss(self,logits,label):
        num_labels = logits.shape[-1]
        loss_object = loss_fn(logits,label)
        return loss_object

    def validation_step(self, batch, batch_idx):
        output = self.model(
            input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        # output = self.model(input_ids=batch['input_ids'], labels=batch['labels'])
        # acc = self.comput_metrix(output.logits, batch['labels'])
        logits = output.logits
        loss = self.countLoss(logits,batch['labels'].float())
        self.log('val_loss', loss,on_epoch=True,prog_bar=True,logger=True)
        # self.log('val_acc', acc)

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        paras = list(
            filter(lambda p: p[1].requires_grad, self.named_parameters()))
        paras = [{
            'params':
            [p for n, p in paras if not any(nd in n for nd in no_decay)],
            'weight_decay': self.args.weight_decay
        }, {
            'params': [p for n, p in paras if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }]
        optimizer = torch.optim.AdamW(paras, lr=self.args.learning_rate)
        #optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(paras, lr=self.args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(self.total_step * self.args.warmup),
            self.total_step)

        return [{
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }]

    
if __name__ == '__main__':
    pass