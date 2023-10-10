import time
import torch
from transformers.optimization import get_linear_schedule_with_warmup
from tqdm import tqdm
import wandb
def forwardModel(batch,device,model,loss_fn):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    #print(labels.shape)
    output = model(input_ids = input_ids, attention_mask = attention_mask,labels=labels)
            
    #print(logits.shape,labels.shape)
    #loss = loss_fn(logits, labels)
    logits = output.logits
    loss = output.loss
    return logits,loss

def validateEpoch(valid_loader,device,model,loss_fn,logger):
    loss_valid = 0
    model.eval()
    for step , batch in enumerate(valid_loader):
        logits, loss = forwardModel(batch,device,model,loss_fn)

        loss_valid+=loss.item()

    logger.info(f" This epoch VALID LOSS {loss_valid/len(valid_loader)}")
    return loss_valid/len(valid_loader)
    

def doTrain(hp,model,train_loader,valid_loader,loss_fn,logger,device,name='test'):
    steps_epoch = len(train_loader)
    num_training_steps = len(train_loader)*hp.data.train_batchsize
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.model.learning_rate)



    #start train
    global_step = 0
    logging_step = 50
    save_step = 10000
    for epoch in range(hp.trainer.max_epochs):
        logger.info("\n=====start training of %d epochs=====" % epoch)
        tic_epoch = time.time()
        model.train().to(device)

        train_data_loader = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, ncols=100)
        loss_epoch = 0
        for step , batch in enumerate(train_data_loader):
            logits, loss = forwardModel(batch,device,model,loss_fn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #lr_scheduler.step()
            loss_item = loss.cpu().item()
            loss_epoch+=loss_item
            global_step+=1

            # 使用wandb记录损失
            #wandb.log({"train_loss_step": loss_item/hp.data.train_batchsize})
            # 更新tqdm进度条
            train_data_loader.set_postfix({"loss": loss.item()}, refresh=True)
        toc_epoch = time.time()
        logger.info(f"Epoch {epoch} took {toc_epoch - tic_epoch:.2f} seconds")
        # 使用wandb记录损失
        #wandb.log({"train_loss_epoch": loss_epoch/(len(train_loader)*hp.data.train_batchsize)})
        val_loss = validateEpoch(valid_loader,device,model,loss_fn,logger)
        if (epoch+1)%10==0:
            torch.save(model,f'outputs/model/{name}-{epoch}-{val_loss}.model')


