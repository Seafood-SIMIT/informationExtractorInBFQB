import time
import torch
from transformers.optimization import get_linear_schedule_with_warmup
from tqdm import tqdm
import wandb
def doTrain(hp,model,train_loader,valid_loader,loss_fn,logger,device):
    steps_epoch = len(train_loader)
    num_training_steps = len(train_loader)*hp.data.train_batchsize
    
    decay_params = [p for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]
    optimizer = torch.optim.AdamW(decay_params, lr=hp.model.learning_rate)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer,
                                                   num_warmup_steps=int(num_training_steps*hp.model.warmup_ratio),
                                                   num_training_steps=num_training_steps)


    #start train
    global_step = 0
    logging_step = 50
    save_step = 10000
    tic_train = time.time()
    for epoch in range(hp.trainer.max_epochs):
        logger.info("\n=====start training of %d epochs=====" % epoch)
        tic_epoch = time.time()
        model.train().to(device)

        train_data_loader = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, ncols=100)
        loss_epoch = 0
        for step , batch in enumerate(train_data_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            logits = model(input_ids = input_ids, attention_mask = attention_mask).logits
            
            optimizer.zero_grad()
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            loss_item = loss.cpu().item()
            loss_epoch+=loss_item
            global_step+=1

            # 使用wandb记录损失
            wandb.log({"train_loss_step": loss_item/hp.data.train_batchsize})
            # 更新tqdm进度条
            train_data_loader.set_postfix({"loss": loss.item()}, refresh=True)
        toc_epoch = time.time()
        logger.info(f"Epoch {epoch} took {toc_epoch - tic_epoch:.2f} seconds")
        # 使用wandb记录损失
        wandb.log({"train_loss_epoch": loss_epoch/(len(train_loader)*hp.data.train_batchsize)})


