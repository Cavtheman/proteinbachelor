from Bio import SeqIO
from string import ascii_uppercase
import numpy as np
import torch
import torch.nn.utils.rnn as rnn

import re
from torch.utils import data
import torch.nn.functional as F


class Dataset(data.Dataset):
    '''
    Checks whether a given sequence is legal

    Parameters
    ----------
    seq : String
        Raw unprocessed string from the fasta file

    Returns
    ----------
    Bool
    '''
    def __is_legal_seq__(self, seq):
        len_val = not (len(seq) > self.max_seq_len)
        cont_val = not(('X' in seq) or ('B' in seq) or ('Z' in seq) or ('J' in seq))
        return len_val and cont_val

    '''
    Generates a dictionary given a string with all the elements that will be in the dictionary

    Parameters
    ----------
    acids : String
        An "alphabet" to generate the dictionary from.
        The last char will be used as a padding value

    Returns
    ----------
    acid_dict : Dictionary of Tensors
        A dictionary with the same length as the input string.
        It is effectively a one-hot encoding of acids
    '''
    def __gen_acid_dict__(self, acids):
        acid_dict = {}
        int_acid_dict = {}
        for i, elem in enumerate(acids):
            temp = torch.zeros(len(acids))
            temp[i] = 1
            acid_dict[elem] = temp
            int_acid_dict[temp] = i
        return acid_dict, int_acid_dict

    '''
    Initialisation for the Dataset

    Parameters
    ----------
    filename : String
        Path to a fasta file with data.
    max_seq_len : Int
        An integer representing the longest sequences we want to take into account.
    acids : String
        An "alphabet" to generate the dictionary from.

    Variables
    ----------
    acid_dict : Dictionary of Tensors
        See __gen_acid_dict__
    data : List of Strings
        The entire input file loaded as strings
    '''
    def __init__(self, filename, max_seq_len, output_type="onehot", acids="ACDEFGHIKLMNPQRSTVWY-", get_prot_class=False):
        elem_list = []
        label_list = []
        self.acids = acids
        self.get_prot_class = get_prot_class
        self.output_type = output_type
        self.acid_dict, self.int_acid_dict = self.__gen_acid_dict__(acids)
        self.max_seq_len = max_seq_len
        # Loading the entire input file into memory
        prot_class_re = re.compile(r" (\w)\.\d+")
        for i, elem in enumerate(SeqIO.parse(filename, "fasta")):
            if self.__is_legal_seq__(elem.seq.upper()):
                elem_list.append(elem.seq.upper())
                if get_prot_class:
                    label_list.append(prot_class_re.search(elem.description).group(1))
        self.data = elem_list
        self.prot_labels = label_list

    '''
    Method to get the length of the dataset

    Returns
    ----------
    Int : Length of the entire dataset
    '''
    def __len__(self):
        return len(self.data)

    '''
    Preprocesses a sequence into something usable by an LSTM and outputs it

    Parameters
    ----------
    index : Int
        Index to take the data from

    Returns
    ----------
    tensor_seq : Tensor of size max_seq_len x len(acid_dict)
        The padded, preprocessed Tensor of one-hot encoded acids.
        If output_type="embed" then it will have size max_seq_len
    labels_seq : Tensor of size max_seq_len
        Contains the labels for each element in tensor_seq
        as the correct index of the one-hot encoding
    valid_elems : Int
        Integer value representing the length of the sequence before padding
    '''
    def __getitem__(self, index):
        seq = self.data[index]
        #print(seq)
        #print(self.acid_dict.keys())
        valid_elems = min(len(seq), self.max_seq_len)

        seq = str(seq).ljust(self.max_seq_len+1, self.acids[-1])
        temp_seq = [self.acid_dict[x] for x in seq]
        if self.output_type == "embed":
            tensor_seq = torch.argmax(torch.stack(temp_seq[:-1]), dim=1).long()
        else:
            tensor_seq = torch.stack(temp_seq[:-1], dim=0).float()#.view(self.max_seq_len, 1, -1)

        # Labels consisting of the index of correct class
        #                                               |
        #                                   CHANGE THIS V TO 1: WHEN FINISHED PREDICTING IDENTITY
        labels_seq = torch.argmax(torch.stack(temp_seq[1:]), dim=1).long()#.view(-1, 1)
        if self.get_prot_class:
            return tensor_seq, labels_seq, valid_elems, self.prot_labels[index]
        else:
            return tensor_seq, labels_seq, valid_elems
