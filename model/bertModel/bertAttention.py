import torch.nn as nn
import torch

class BERTAttention(nn.Module):
    def __init__(self, object_model,hidden_size,num_classes,num_relations,num_heads):
        super(BERTAttention, self).__init__()

        self.object_model = object_model
        self.attention_layer = nn.MultiheadAttention(embed_dim=num_heads,num_heads=num_heads)

        #classifier
        self.lstm = nn.LSTM(num_heads, hidden_size, batch_first = True)
        self.linear_relation = nn.Linear(hidden_size, num_classes)
        self.linear_anchor = nn.Linear(hidden_size, num_relations)

    

    def forward(self, input_ids):
        #print(input_ids.shape)
        # [b,max_length]
        
        logits = self.object_model(input_ids)['logits']
        query =input_ids.unsqueeze(2).float()
        key = logits[:,:,0:1]
        value = logits[:,:,6:7]
        #print(query.shape,key.shape,value.shape)
        attention_output, _ = self.attention_layer(query,key,value)


        classify_input = attention_output * query.int()

        lstm_output, _ = self.lstm(classify_input)

        #relation
        predict_relation = self.linear_relation(lstm_output[:,-1,:])
        predict_anchor = self.linear_anchor(lstm_output[:,-1,:])

        
        return logits,predict_anchor,predict_relation



if __name__ == '__main__':
    from transformers import AutoTokenizer,BertForTokenClassification
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    bert_model = BertForTokenClassification.from_pretrained('bert-base-chinese',num_labels=12)

    model = BERTAttention(bert_model,
                          max_length=128,
                          hidden_size=128,
                          num_classes=13,
                          num_relations = 32,num_heads=1)
    
    fake_input = torch.randint(1,10,(2,128))
    logits,predict_anchor,predict_relation = model(fake_input)
    print(f"logits shape:{logits.shape} predict_anchor.shape{predict_anchor.shape}  predict_relation.shape{predict_relation.shape}")
    #print(output.shape)
    #torch.Size([2, 128, 3])