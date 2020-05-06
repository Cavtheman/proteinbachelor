from Bio import SeqIO
from string import ascii_uppercase
import numpy as np
import torch
import torch.nn.utils.rnn as rnn

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
    def __init__(self, filename, max_seq_len, acids="ACDEFGHIKLMNPQRSTVWY-"):
        elem_list = []
        self.acids = acids
        self.acid_dict, self.int_acid_dict = self.__gen_acid_dict__(acids)
        self.max_seq_len = max_seq_len
        # Loading the entire input file into memory
        for i, elem in enumerate(SeqIO.parse(filename, "fasta")):
            if self.__is_legal_seq__(elem.seq):
                elem_list.append(elem.seq)
        self.data = elem_list

    '''
    Method to get the length of the dataset

    Returns
    ----------
    Int : Length of the entire dataset
    '''
    def __len__(self):
        return len(self.data)

    '''
    Preprocesses a sequence into something usable by an LSTM

    Parameters
    ----------
    seq : String
        The sequence to be prepared

    Returns
    ----------
    tensor_seq : Tensor of size max_seq_len x len(acid_dict)
        The padded, preprocessed Tensor of one-hot encoded acids
    labels_seq : Tensor of size max_seq_len
        Contains the labels for each element in tensor_seq
        as the correct index of the one-hot encoding
    valid_elems : Int
        Integer value representing the length of the sequence before padding
    '''
    def __prepare_seq__(self, seq):
        valid_elems = min(len(seq), self.max_seq_len)

        seq = str(seq).ljust(self.max_seq_len+1, self.acids[-1])
        temp_seq = [self.acid_dict[x] for x in seq]
        tensor_seq = torch.stack(temp_seq[:-1], dim=0).float()#.view(self.max_seq_len, 1, -1)

        # Labels consisting of the index of correct class
        #                                               I
        #                                   CHANGE THIS V TO 1: WHEN FINISHED PREDICTING IDENTITY
        labels_seq = torch.argmax(torch.stack(temp_seq[:-1]), dim=1).long()#.view(-1, 1)
        return tensor_seq, labels_seq, valid_elems

    '''
    Used by the data_generator class to get items

    Parameters
    ----------
    index : Int
        Index to take the data from

    Returns
    ----------
    Tuple : See __prepare_seq__
        A prepared sequence from a specific index in the dataset
    '''
    def __getitem__(self, index):
        return self.__prepare_seq__(self.data[index])
