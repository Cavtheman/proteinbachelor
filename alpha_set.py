import torch
import torch.nn.utils.rnn as rnn
from torch.utils import data

class alpha_set(data.Dataset):
    def __gen_acid_dict__(self, acids):
        acid_dict = {}
        for i, elem in enumerate(acids):
            temp = torch.zeros(len(acids))
            temp[i] = 1
            acid_dict[elem] = temp
        return acid_dict

    def gen_alphabet(self, mod_val):
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        index = mod_val % 26
        return alphabet[index:] + alphabet[:index]

    def __init__(self, acids, length, num_seqs):
        self.max_seq_len = length
        self.acids = acids
        self.acid_dict = self.__gen_acid_dict__(acids)
        self.data = [self.gen_alphabet(i) for i in range(num_seqs)]

    def __prepare_seq__(self, seq):
        valid_elems = min(len(seq), self.max_seq_len)
        seq = str(seq).ljust(self.max_seq_len+1, '-')
        temp_seq = [self.acid_dict[x] for x in seq]
        tensor_seq = torch.stack(temp_seq[:-1]).float()
        #valid_elems = torch.Tensor([elem != '-' for elem in seq[:-1]])

        # Labels consisting of the raw tensor
        # labels_seq = torch.stack(temp_seq[1:]).long()

        # Label consisting of last element
        # labels_seq = temp_seq[-1].long()

        # Labels consisting of the index of correct class
        labels_seq = torch.argmax(torch.stack(temp_seq[1:]), dim=1).long()

        #print(labels_seq.size())
        #print(tensor_seq.size())
        #labels_seq = torch.transpose(labels_seq, 0, 1)
        #tensor_seq = torch.transpose(tensor_seq, 0, 1)
        #print("Seq shape:", tensor_seq[1:].size())
        return tensor_seq, labels_seq, valid_elems

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.__prepare_seq__(self.data[index])
