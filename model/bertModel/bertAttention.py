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

        self.classifier = nn.Linear(in_features=max_seq_length,out_features=num_anchor*(3+num_relation))
    

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
        classifier_out = self.classifier(lstm_output)
        x = classifier_out.view(classifier_out.size(0), self.anchor_num,3+self.num_relation)

        for i in range(3):
            x[:,:,i] = torch.sigmoid(x[:,:,i])
        predict_anchor = x[:,:,0:3]
        predict_relation = torch.softmax(x[:,:,3:],dim=2)

        
        return {'predict_entity':logits,
                'predict_anchor':predict_anchor,
                'predict_relation':predict_relation,
                'loss_bert':loss}



if __name__ == '__main__':
    from transformers import AutoTokenizer,BertForTokenClassification
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    bert_model = BertForTokenClassification.from_pretrained('bert-base-chinese',num_labels=12)

    model = BERTAttention(object_model = bert_model,max_seq_length = 256,num_heads =1,num_anchor = 5,num_relation = 32)
    
    fake_input = torch.randint(1,10,(2,128))
    labels = torch.zeros((2,128),dtype = torch.long)
    output = model(fake_input,fake_input,labels = labels)

    print(f"logits shape:{output['predict_entity'].shape} predict_anchor.shape{output['predict_anchor'].shape}  predict_relation.shape{output['predict_relation'].shape}, loss{output['loss_bert'].shape}")
    #print(output.shape)
    #torch.Size([2, 128, 3])