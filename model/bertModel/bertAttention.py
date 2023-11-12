import torch.nn as nn
import torch

class BERTAttention(nn.Module):
    def __init__(self, object_model,max_seq_length,num_heads=2,num_anchor=10,num_relation=56):
        super(BERTAttention, self).__init__()

        self.max_seq_length = max_seq_length
        self.bert_model = object_model
        self.attention_layer = nn.MultiheadAttention(embed_dim=num_heads,num_heads=num_heads)

        #classifier
        self.lstm = nn.LSTM(num_heads, max_seq_length, batch_first = True)
        self.anchor_num = num_anchor

        self.num_relation = num_relation

        self.fc = nn.Linear(in_features=max_seq_length,out_features=num_anchor*(3+num_relation))
        self.conv = nn.Sequential(
            #cnn1hp.signal.wavelet_energyfeatures
            nn.Conv1d(1, 32, kernel_size=1, stride=1,padding=1),#nx48x29
            nn.Conv1d(32, 32, kernel_size=1, stride=1,padding=1),#nx48x29
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=1, stride=1,padding=1),#nx64x6
            nn.Conv1d(64, 64, kernel_size=1, stride=1,padding=1),#nx64x6
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64,64, kernel_size=1, stride=1,padding=1),#nxclass_numx1
            nn.Conv1d(64, 64, kernel_size=1, stride=1,padding=1),#nxclass_numx1
            )
        self.fc1 = nn.Sequential(
            nn.Linear(64*412,1024),
            nn.Linear(1024,out_features=(3+self.num_relation)*num_anchor),
            #nn.BatchNorm1d(3),
            #nn.Sigmoid(),
        ) 
    

    def forward(self, input_ids, attention_mask,labels = None):
        #print(input_ids.shape)
        # [b,max_length]
        def prt(data):
            print(torch.sum(data[0] - data[1]))
        loss = None
        output = self.bert_model(input_ids = input_ids, attention_mask = attention_mask,labels=labels)
        logits = output['logits']
        if labels != None:
            loss = output['loss']

        query =input_ids.unsqueeze(2).float()
        logits_argmax = torch.argmax(logits,dim=2).clone().detach().unsqueeze(2)
        #print(query.shape,query.dtype,logits_argmax.shape,logits_argmax.dtype)
        bert_output = torch.cat([query,logits_argmax],dim=2)

        lstm_output, _ = self.lstm(bert_output)

        #relation
        lstm_output = lstm_output[:,-1,:].unsqueeze(1)
        #print(lstm_output.shape) #[b,1,256]
        classifier_out = self.conv(lstm_output)
        x = classifier_out.reshape(classifier_out.size(0),-1)
        anchor_param = self.fc1(x)
        anchor_param = anchor_param.reshape(classifier_out.size(0), self.anchor_num,-1)

        predict_anchor = anchor_param[:,:,0:2]
        predict_conf = anchor_param[:,:,2:3]
        predict_relation = anchor_param[:,:,3:]
        #predict_relation = torch.softmax(anchor_relation,dim=2)
        #print(predict_relation.shape)
        #print(predict_relation[:,:,0:5])

        
        return {'predict_entity':logits,
                'predict_anchor':predict_anchor,
                'predict_conf': predict_conf,
                'predict_relation':predict_relation,
                'loss_bert':loss}



if __name__ == '__main__':
    from transformers import AutoTokenizer,BertForTokenClassification
    #tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    #bert_model = BertForTokenClassification.from_pretrained('bert-base-chinese',num_labels=12)
    model_dir = 'bert-base-chinese'
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    bert_model = BertForTokenClassification.from_pretrained(model_dir,num_labels=12)

    model = BERTAttention(object_model = bert_model,max_seq_length = 400,num_heads =1,num_anchor = 10,num_relation = 56)
    
    fake_input = torch.randint(1,10,(2,128))
    labels = torch.zeros((2,128),dtype = torch.long)
    output = model(fake_input,fake_input,labels = labels)

    print(f"logits shape:{output['predict_entity'].shape} predict_anchor.shape{output['predict_anchor'].shape}  predict_relation.shape{output['predict_relation'].shape}, loss{output['loss_bert'].shape}")
    #print(output.shape)
    #torch.Size([2, 128, 3])