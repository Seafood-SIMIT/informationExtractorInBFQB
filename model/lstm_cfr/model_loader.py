from .lstmCFR import BiLSTMCRFModel
from transformers import AutoTokenizer
import torch
def loadModel(args,num_labels):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    print("num_labels被设置为",num_labels)
    model = BiLSTMCRFModel(vocab_size=tokenizer.vocab_size, num_tags=num_labels,
                           embedding_dim=args.max_seq_length,
                           hidden_dim=args.hidden_dim,
                           lstm_layers=args.lstm_layers)
    return tokenizer,model

if __name__=='__main__':
    hp = HParam('/Users/sunlin/Documents/workdir/ieer/infromationExtractorInBFQB-main/config/default.yaml')
    num_labels = hp.data.num_labels
    tokenizer, model = loadModel(hp.model,num_labels)

    text = '我方四只企鹅攻击了敌方指挥部'
        # 分词并进行预测
    prompt_ids = tokenizer.encode_plus(text,max_length=128,
                                                padding='max_length',
                                                 add_special_tokens=True,
                                                 return_attention_mask=True,
                                                 return_tensors='pt')
    outputs = model(**prompt_ids)

    # 获取预测的标签概率分布
    predictions = outputs.logits
    #应为2，136，128
    print(predictions.shape)
    predictions_probs = torch.nn.functional.softmax(predictions, dim=2)
    print(predictions_probs)
