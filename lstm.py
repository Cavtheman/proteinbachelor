
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F

class LSTM_model(nn.Module):
    '''
    Initialises the model for use

    Parameters
    ----------
    input_size : Int
        Length of the "alphabet".
    hidden_layer_size : Int
        Size of the hidden layer between the LSTM and the linear layer.
    nr_hidden_layers: Int
        Number of LSTMs in sequence.
    max_seq_len : Int
        An integer representing the longest sequences we want to take into account.
    batch_size : Int
        The number of sequences the model will look at in parallel at any time

    Variables
    ----------
    model : nn.LSTM
        The actual pytorch LSTM to be trained
    linear : nn.Linear
        A single linear layer for converting the LSTMs output to the same size as the input
    '''
    def __init__(self, input_size, hidden_layer_size, nr_hidden_layers, max_seq_len, batch_size, processor):
        super(LSTM_model, self).__init__()
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.nr_hidden_layers = nr_hidden_layers
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.init_hidden = (torch.randn(self.nr_hidden_layers,
                                        self.batch_size,
                                        self.hidden_layer_size).to(processor),
                            torch.randn(self.nr_hidden_layers,
                                        self.batch_size,
                                        self.hidden_layer_size).to(processor))

        self.model = nn.LSTM(input_size, hidden_layer_size, nr_hidden_layers, batch_first=False)

        self.linear = nn.Linear(hidden_layer_size, input_size)

    '''
    Forward pass of the model

    Parameters
    ----------
    input_data : PackedSequence with batch_size batches and input_size features
        The input data to be predicted/trained on

    Returns
    ----------
    tag_space : Tensor of size max_seq_len x batch_size x input_size
        Padded output of predictions for the sequences
    lstm_out : PackedSequence with batch_size x hidden_layer_size features
        Hidden layers of the lstm for all elements in the sequences.
        Used for dimensionality reduction later to see differences between proteins
    '''
    def forward(self, input_data):
        lstm_out, (hn, cn) = self.model(input_data, self.init_hidden)

        lin_in, _ = rnn.pad_packed_sequence(lstm_out, total_length=self.max_seq_len)

        tag_space = self.linear(lin_in.view(-1,lin_in.size()[2]))
        tag_space = tag_space.view(self.max_seq_len, self.batch_size, -1)
        #tag_space = self.linear(lin_in)

        #print(lin_in.size())
        #print(lin_in.view(-1,lin_in.size()[2]).size())
        #print(tag_space.size())

        return tag_space, lstm_out

'''
Not in use any more
'''
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
