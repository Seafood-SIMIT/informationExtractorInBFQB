import torch
import pytorch_lightning as pl
from transformers.optimization import get_linear_schedule_with_warmup
import torch.nn.CrossEntropyLoss as CELoss
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
            input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
        # output = self.model(input_ids=batch['input_ids'], labels=batch['labels'])
        # acc = self.comput_metrix(output.logits, batch['labels'])
        logits = output.logits
        loss = self.countLoss(logits,batch['labels'])
        self.log('train_loss', loss,on_epoch=True, prog_bar=True,logger=True)
        return output.loss

    def countLoss(logits,label):
        is_object = logits[0:3]
        is_shuxing = logits[3:6]
        is_relation = logits[6:8]
        is_toid = logits[8:]
        loss_object = CELoss(is_object,label[0])
        return loss_object
    def comput_metrix(self, logits, labels):
        y_pred = torch.argmax(logits, dim=-1)
        y_pred = y_pred.view(size=(-1,))
        y_true = labels.view(size=(-1,)).float()
        corr = torch.eq(y_pred, y_true)
        acc = torch.sum(corr.float()) / labels.size()[0]
        return acc

    def validation_step(self, batch, batch_idx):
        output = self.model(
            input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
        # output = self.model(input_ids=batch['input_ids'], labels=batch['labels'])
        # acc = self.comput_metrix(output.logits, batch['labels'])
        self.log('val_loss', output.loss,on_epoch=True,prog_bar=True,logger=True)
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