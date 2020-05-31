
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
    def __init__(self, input_size=23, embed_size=10,  hidden_layer_size=512, nr_hidden_layers=1, max_seq_len=500, batch_size=64, processor="cuda", bidir=False, dropout=0):
        super(LSTM_model, self).__init__()
        self.input_size = input_size
        self.embed_size = embed_size
        self.hidden_layer_size = hidden_layer_size
        self.nr_hidden_layers = nr_hidden_layers
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.num_dir = 2 if bidir else 1
        self.processor = processor
        self.dropout = dropout
        self.model = nn.LSTM(embed_size, hidden_layer_size, nr_hidden_layers, batch_first=False, dropout=dropout, bidirectional=bidir)

        self.embed = nn.Embedding(input_size, embed_size)

        self.linear = nn.Linear(hidden_layer_size*self.num_dir, input_size)


    def init_hidden(self):

        return (torch.randn(self.nr_hidden_layers*self.num_dir,
                            self.batch_size,
                            self.hidden_layer_size).to(self.processor),
                torch.randn(self.nr_hidden_layers*self.num_dir,
                            self.batch_size,
                            self.hidden_layer_size).to(self.processor))
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
    def forward(self, input_data, valid_elems):
        embedding = self.embed(input_data)
        packed = rnn.pack_padded_sequence(embedding, valid_elems, enforce_sorted=False)

        lstm_out, (hn, cn) = self.model(packed, self.init_hidden())
        lin_in, _ = rnn.pad_packed_sequence(lstm_out, total_length=self.max_seq_len)

        tag_space = self.linear(lin_in.view(-1,lin_in.size()[2]))
        tag_space = tag_space.view(self.max_seq_len, lin_in.size()[1], -1)

        return tag_space, lin_in #lstm_out

    def save(self, filename):
        args_dict = {
            "input_size": self.input_size,
            "embed_size": self.embed_size,
            "hidden_layer_size": self.hidden_layer_size,
            "nr_hidden_layers": self.nr_hidden_layers,
            "max_seq_len": self.max_seq_len,
            "batch_size": self.batch_size,
            "processor": self.processor,
            "bidir": True if self.num_dir == 2 else False,
            "dropout": self.dropout
        }
        torch.save({
            "state_dict": self.state_dict(),
            "args_dict": args_dict
        }, filename) #model.state_dict(), "temp_best_model.pth")
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
