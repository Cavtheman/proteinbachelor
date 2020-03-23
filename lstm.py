import torch
import torch.nn as nn

class LSTM_model(nn.Module):
    def __init__(self, input_size, hidden_layer_size, nr_hidden_layers, max_seq_len):
        super(LSTM_model, self).__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.nr_hidden_layers = nr_hidden_layers

        self.model = nn.LSTM(input_size, hidden_layer_size, nr_hidden_layers, batch_first=True)

        self.linear = nn.Linear(hidden_layer_size, input_size)

    def forward(self, input_data):
        lstm_out, (hn, cn) = self.model(input_data)
        #print("LSTM out:", lstm_out.size())
        tag_space = self.linear(lstm_out)
        #print("Tag Space:", tag_space.size())
        #tag_scores = F.log_softmax(tag_space, dim=1)
        #print(type(tag_scores))
        #output = torch.argmax(tag_space, dim=2).float()
        return tag_space, (hn, cn)
