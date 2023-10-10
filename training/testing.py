from tqdm import tqdm
from .training import forwardModel
import numpy as np
def labelDecode(text,logits):

    ans = np.argmax(logits,axis=1)
    # shiti start
    print(text)
    print('时间:',"".join([text[i] for i in np.where(ans==1)[0]]+
                         [text[i] for i in np.where(ans==2)[0]]+
                         [text[i] for i in np.where(ans==3)[0]]))
    print('actor1:',"".join([text[i] for i in np.where(ans==4)[0]]+
                         [text[i] for i in np.where(ans==5)[0]]+
                         [text[i] for i in np.where(ans==6)[0]]))
    print('actor2:',"".join([text[i] for i in np.where(ans==7)[0]]+
                         [text[i] for i in np.where(ans==8)[0]]+
                         [text[i] for i in np.where(ans==9)[0]]))
    print('地点:',"".join([text[i] for i in np.where(ans==10)[0]]+
                         [text[i] for i in np.where(ans==11)[0]]+
                         [text[i] for i in np.where(ans==12)[0]]))
    print('='*20)
    
        #print(text[i],'\t',shiti_s[i]>0.1)
    #shiti_o = shiti[]
    

def doTest(hp,model,test_loader,loss_fn,logger,device):
    model.to(device)
    model.eval()
    for step , batch in enumerate(test_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        logits = model(input_ids = input_ids, attention_mask = attention_mask).logits

        labelDecode(batch['text'][0],logits[0].cpu().detach().numpy())
        break
