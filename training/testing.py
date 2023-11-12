from tqdm import tqdm
from .training import forwardModel
import numpy as np
from dataloader.duie.data_decode import labelDecode

def doTest(hp,model,test_loader,loss_fn,logger,device):
    model.to(device)
    model.eval()
    for step , batch in enumerate(test_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        predicts = model(input_ids = input_ids, attention_mask = attention_mask)

        labelDecode(batch['text'],predicts)
        break
