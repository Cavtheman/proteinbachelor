import torch

'''
Prints out human-readable sequences, from the output of the LSTM

Parameters
----------
preds : Tensor of size max_seq_len x batch_size x input_size
    This can come directly from the LSTM to be converted
valid : Int
    Integer value representing the length of the sequence before padding
'''
def print_seq(preds, valid, alphabet):
    for i, seq in enumerate(preds):
        print("Sequence {}".format(i))
        indexes = torch.argmax(seq[:valid[i]], dim=1)
        ret_val = [alphabet[x] for x in indexes]
        print("".join(ret_val))
