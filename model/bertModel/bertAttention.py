import torch.nn as nn
import torch

class BERTAttention(nn.Module):
    def __init__(self, object_model,max_seq_length,num_heads,num_anchor,num_relation):
        super(BERTAttention, self).__init__()

        self.max_seq_length = max_seq_length
        self.object_model = object_model
        self.attention_layer = nn.MultiheadAttention(embed_dim=num_heads,num_heads=num_heads)

        #classifier
        self.lstm = nn.LSTM(num_heads, max_seq_length, batch_first = True)
        self.anchor_num = num_anchor

        self.num_relation = num_relation

        self.fc = nn.Linear(in_features=max_seq_length,out_features=num_anchor*(3+num_relation))
        self.conv = nn.Sequential(
            #cnn1hp.signal.wavelet_energyfeatures
            nn.Conv1d(1, 32, kernel_size=5, stride=2,bias=True),#nx48x29
            nn.Conv1d(32, 32, kernel_size=1, stride=1,bias=True),#nx48x29
            #nn.LayerNorm(32),
            nn.LeakyReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2,bias=True),#nx64x6
            nn.Conv1d(64, 64, kernel_size=1, stride=1,bias=True),#nx64x6
            #nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Conv1d(64, 8, kernel_size=4, stride=1,bias=True),#nxclass_numx1
            nn.Conv1d(8, 8, kernel_size=4, stride=1,bias=True),#nxclass_numx1
            nn.ReLU(),
            )
        self.fc1 = nn.Sequential(
            nn.Linear(736,1024),
            #nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(1024,out_features=num_anchor*(3+num_relation)),
            #nn.BatchNorm1d(3),
            nn.Sigmoid(),
        ) 
    

    def forward(self, input_ids, attention_mask,labels = None):
        #print(input_ids.shape)
        # [b,max_length]
        
        loss = None
        output = self.object_model(input_ids = input_ids, attention_mask = attention_mask,labels=labels)
        logits = output['logits']
        loss = output['loss']

        query =input_ids.unsqueeze(2).float()
        key = logits[:,:,0:1]
        value = logits[:,:,6:7]
        #print(query.shape,key.shape,value.shape)
        attention_output, _ = self.attention_layer(query,key,value)


        classify_input = attention_output * query.int()

        lstm_output, _ = self.lstm(classify_input)

        #relation
        #print(lstm_output[:,-1,:].shape)
        lstm_output = lstm_output[:,-1,:].unsqueeze(1)
        #print(lstm_output.shape) #[b,1,256]
        classifier_out = self.conv(lstm_output)
        x = classifier_out.view(classifier_out.size(0),-1)
        x = self.fc1(x)
        x = x.view(classifier_out.size(0), self.anchor_num,3+self.num_relation)

        #for i in range(3):
        #    x[:,:,i] = torch.sigmoid(x[:,:,i])
        #predict_anchor = x[:,:,0:3]
        #predict_relation = x[:,:,3:]

        
        return {'predict_entity':logits,
                'predict_anchor':predict_anchor,
                'predict_relation':predict_relation,
                'loss_bert':loss}



if __name__ == '__main__':
    from transformers import AutoTokenizer,BertForTokenClassification
    #tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    #bert_model = BertForTokenClassification.from_pretrained('bert-base-chinese',num_labels=12)
    model_dir = '/root/autodl-tmp/bert/bert-base-chinese'
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    bert_model = BertForTokenClassification.from_pretrained(model_dir,num_labels=12)

    model = BERTAttention(object_model = bert_model,max_seq_length = 400,num_heads =1,num_anchor = 10,num_relation = 56)
    
    fake_input = torch.randint(1,10,(2,128))
    labels = torch.zeros((2,128),dtype = torch.long)
    output = model(fake_input,fake_input,labels = labels)

    print(f"logits shape:{output['predict_entity'].shape} predict_anchor.shape{output['predict_anchor'].shape}  predict_relation.shape{output['predict_relation'].shape}, loss{output['loss_bert'].shape}")
    #print(output.shape)
    #torch.Size([2, 128, 3])