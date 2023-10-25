import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from TorchCRF import CRF
class BiLSTMCRFModel(nn.Module):
    def __init__(self, vocab_size, num_tags, embedding_dim, hidden_dim, lstm_layers=1):
        super(BiLSTMCRFModel, self).__init__()

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # BiLSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=lstm_layers, bidirectional=True, batch_first=True)

        # Linear layer to project the output of BiLSTM to the number of tags
        self.hidden2tag = nn.Linear(hidden_dim * 2, num_tags)

        # CRF layer
        self.crf = CRF(num_tags)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)

        # Pass the embedded input through the LSTM layer
        lstm_out, _ = self.lstm(embedded)

        # Project the output of BiLSTM to the number of tags
        emissions = self.hidden2tag(lstm_out)

        return emissions


if __name__ == '__main__':
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')

    model = BiLSTMCRFModel(tokenizer.vocab_size,
                        num_tags=3,
                        embedding_dim=128,
                        hidden_dim=128,
                        lstm_layers=2)
    
    fake_input = torch.zeros((2,128),dtype=torch.int32)
    output = model(fake_input)
    print(output.shape)
    #torch.Size([2, 128, 3])
