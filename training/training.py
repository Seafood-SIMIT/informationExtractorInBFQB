import time
import torch
import torch.nn as nn
from transformers.optimization import get_linear_schedule_with_warmup
from tqdm import tqdm
import wandb

def forwardModel(batch,device,model,loss_fn):
    lossfn_anchor = nn.MSELoss()
    lossfn_relation = nn.CrossEntropyLoss()
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    label_entities = batch['label_entities'].to(device)
    label_anchor = batch['label_anchor'].to(device)
    label_relations = batch['label_relations'].to(device)

    #print(labels.shape)
    output = model(input_ids = input_ids, attention_mask = attention_mask,labels=label_entities)
            
    #print(output['predict_anchor'].shape,label_anchor.shape,output['predict_relation'].shape,label_relations.shape)
    #loss = output['loss_bert'] + lossfn_anchor(output['predict_anchor'],label_anchor) + lossfn_relation(output['predict_relation'].view(-1,output['predict_relation'].shape[-1]),label_relations.view(-1))
    #loss = lossfn_anchor(output['predict_anchor'],label_anchor) 
    #loss = lossfn_relation(output['predict_relation'].view(-1,output['predict_relation'].shape[-1]),label_relations.view(-1))
    loss_relation = 0
    for i in range(output['predict_relation'].size(1)):
        single_realtion = output['predict_relation'][:,i,:]
        loss_relation+=lossfn_relation(single_realtion,label_relations[:,i])
        
    loss = loss_relation/output['predict_relation'].size(1) + output['loss_bert'] + lossfn_anchor(output['predict_anchor'],label_anchor)
    return output['predict_entity'], output['predict_anchor'],output['predict_relation'],loss 

def validateEpoch(valid_loader,device,model,loss_fn,logger):
    loss_valid = 0
    model.eval()
    for step , batch in enumerate(valid_loader):
        entities_predict, anchor_predict,relation_predict,loss  = forwardModel(batch,device,model,loss_fn)

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
            entities_predict, anchor_predict,relation_predict,loss  = forwardModel(batch,device,model,loss_fn)
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
        #logger.info(f"Epoch {epoch} took {toc_epoch - tic_epoch:.2f} seconds")
        logger.info(f"Epoch {epoch} Loss {loss_epoch/len(train_data_loader)}")
        # 使用wandb记录损失
        if not hp.trainer.debug_mode:
            #wandb.log({"train_loss_epoch": loss_epoch/(len(train_loader)*hp.data.train_batchsize)})
            val_loss = validateEpoch(valid_loader,device,model,loss_fn,logger)
            if (epoch)%10==0:
                torch.save(model,f'output/model/{name}-{epoch}-{val_loss:.2f}.model')
                #torch.save(model,f'/root/autodl-tmp/output/model/{name}-{epoch}-{val_loss:.2f}.model')


