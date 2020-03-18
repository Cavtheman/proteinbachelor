from Bio import SeqIO
from string import ascii_uppercase
import numpy as np
import torch
import torch.nn.utils.rnn as rnn

from torch.utils import data
import torch.nn.functional as F


class Dataset(data.Dataset):
    # Checks whether a given sequence is legal
    def __is_legal_seq__(self, seq):
        len_val = not (len(seq) > self.max_seq_len)
        cont_val = not(('X' in seq) or ('B' in seq) or ('Z' in seq) or ('J' in seq))
        return len_val and cont_val

    # Generates a dictionary given a string with all the elements
    def __gen_acid_dict__(self, acids):
        acid_dict = {}
        for i, elem in enumerate(acids):
            temp = torch.zeros(len(acids), dtype=torch.long)
            temp[i] = 1
            acid_dict[elem] = temp
        return acid_dict

    def __init__(self, filename, max_seq_len, acids="ACDEFGHIKLMNPQRSTUVWY-"):
        elem_list = []
        self.acid_dict = self.__gen_acid_dict__(acids)
        self.max_seq_len = max_seq_len

        # Loading the entire input file into memory
        for i, elem in enumerate(SeqIO.parse(filename, "fasta")):
            if self.__is_legal_seq__(elem.seq):
                elem_list.append(elem.seq)
        self.data = elem_list

    def __len__(self):
        return len(self.data)

    def __prepare_seq__(self, seq):
        seq = str(seq).ljust(self.max_seq_len+1, '-')
        tensor_seq = torch.stack([self.acid_dict[x] for x in seq])
        return tensor_seq[:-1].float(), tensor_seq[-1].long()

    def __getitem__(self, index):
        return self.__prepare_seq__(self.data[index])


# Deprecated
class data_generator():
    def __init__(self, filename, max_seq_len, acids="ACDEFGHIKLMNPQRSTUVWY-"):
        self.__acids__ = acids
        self.__parser__ = SeqIO.parse(filename, "fasta")

        self.acid_dict = {}
        self.max_seq_len = max_seq_len
        self.data = torch.Tensor(1, self.max_seq_len, len(self.acid_dict))
        self.targets = torch.Tensor(1, self.max_seq_len, len(self.acid_dict))
        self.gen_acid_dict(acids)

    # If a sequence contains one of XBZJ it will be discarded,
    # and if it is longer than the given max_seq_length
    def __is_legal_seq__(self, seq):
        len_val = not (len(seq) > self.max_seq_len)
        cont_val = not(('X' in seq) or ('B' in seq) or ('Z' in seq) or ('J' in seq))
        return len_val and cont_val

    # Generate a dictionary if none is supplied,
    def gen_acid_dict(self, acids):
        for i, elem in enumerate(acids):
            temp = np.zeros(len(acids))
            temp[i] = 1
            self.acid_dict[elem] = temp
        return self.acid_dict

    # Read num_elems sequences from the given file
    def gen_data(self, num_elems):
        ret_val = []
        targets = []
        i = 0
        for record in self.__parser__:
            if (i == num_elems):
                break
            elif (not self.__is_legal_seq__(record.seq)):
                continue
            else:
                pad_vals = np.full((self.max_seq_len - len(record), len(self.acid_dict)), self.acid_dict['-'])
                seq_vals = np.array([self.acid_dict[elem] for elem in record])
                temp_target_vals = [self.acid_dict[elem] for elem in record]
                temp_target_vals.append(self.acid_dict['-'])
                target_vals = np.array(temp_target_vals[1:])

                ret_val.append(torch.tensor(np.concatenate((seq_vals, pad_vals), axis=0), dtype=torch.float))
                targets.append(torch.tensor(np.concatenate((target_vals, pad_vals), axis=0), dtype=torch.float))
                i += 1

        self.data = torch.stack(ret_val)
        self.targets = torch.stack(targets)
        #self.data = torch.Tensor(num_elems, self.max_seq_len, len(self.acid_dict))
        #torch.cat(ret_val, out=self.data)

'''
large_file = "uniref50.fasta"

data_gen = data_generator(large_file, 2000)
data_gen.gen_data(1000)
#print(data_gen.acid_dict['A'])
print(data_gen.data.size())
#print(data_gen.data[0])

test = torch.tensor([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]],[[13,14,15],[16,17,18]]])
print(test)
print(test.size())
print(test.view(2,3,3))
print(test.view(2,3,3).size())

#print(data_gen.data[0].shape)
'''
