import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

class LSTM_model(nn.Module):
    def __init__(self, input_size, hidden_layer_size, nr_hidden_layers, max_seq_len, batch_size):
        super(LSTM_model, self).__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.nr_hidden_layers = nr_hidden_layers
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

        self.model = nn.LSTM(input_size, hidden_layer_size, nr_hidden_layers, batch_first=False)

        self.linear = nn.Linear(hidden_layer_size, input_size)

    def forward(self, input_data):
        #print(input_data.data.size())
        hidden = (torch.randn(self.nr_hidden_layers, self.batch_size, self.hidden_layer_size),
                  torch.randn(self.nr_hidden_layers, self.batch_size, self.hidden_layer_size))
        lstm_out, (hn, cn) = self.model(input_data)
        lin_in, _ = rnn.pad_packed_sequence(lstm_out, total_length=self.max_seq_len)
        #print("LSTM out:", lstm_out.size())
        tag_space = self.linear(lin_in)
        #print("Tag Space:", tag_space.size())
        #tag_scores = F.log_softmax(tag_space, dim=1)
        #print(type(tag_scores))
        #output = torch.argmax(tag_space, dim=2).float()
        return tag_space, lstm_out

class LSTMCell(nn.Module):
    def __init__(self, feature_size, hidden_layer_size, nr_hidden_layers, max_seq_len):
        super(LSTMCell, self).__init__()
        self.feature_size = feature_size
        self.hidden_layer_size = hidden_layer_size
        self.nr_hidden_layers = nr_hidden_layers
        self.max_seq_len = max_seq_len

        self.model = nn.LSTMCell(feature_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, feature_size)


    def init_hidden(self, batch_size):
        return (torch.randn(batch_size, self.hidden_layer_size),
                torch.randn(batch_size, self.hidden_layer_size))


    def forward(self, input_data, hidden):
        #print(hn, cn)
        lstm_out, _ = self.model(input_data, hidden)
        #print(hn, cn)
        output = self.linear(lstm_out)
        #print(output)
        return output, lstm_out
