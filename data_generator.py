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
            temp = torch.zeros(len(acids))
            temp[i] = 1
            acid_dict[elem] = temp
        return acid_dict

    def __init__(self, filename, max_seq_len, acids="ACDEFGHIKLMNPQRSTVWY-"):
        elem_list = []
        self.acids = acids
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
        temp_seq = [self.acid_dict[x] for x in seq]
        tensor_seq = torch.stack(temp_seq[:-1]).float()

        # Labels consisting of the raw tensor
        # labels_seq = torch.stack(temp_seq[1:]).long()

        # Label consisting of last element
        # labels_seq = temp_seq[-1].long()

        # Labels consisting of the index of correct class
        labels_seq = torch.argmax(torch.stack(temp_seq[1:]), dim=1).long()

        #print("Seq shape:", tensor_seq[1:].size())
        return tensor_seq, labels_seq

    def __getitem__(self, index):
        return self.__prepare_seq__(self.data[index])
