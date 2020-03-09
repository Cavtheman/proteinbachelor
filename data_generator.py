from Bio import SeqIO
from string import ascii_uppercase
import numpy as np
import torch

class data_generator():
    def __init__(self, filename, max_seq_length, acid_dict={}):
        self.__acids__ = "ACDEFGHIKLMNPQRSTVWY-"
        self.__parser__ = SeqIO.parse(filename, "fasta")

        self.acid_dict = acid_dict
        self.data = []
        self.max_seq_length = max_seq_length

        if (acid_dict == {}):
            self.gen_acid_dict()

    # If a sequence contains one of XBZJ it will be discarded,
    # and if it is longer than the given max_seq_length
    def __is_legal_seq__(self, seq):
        len_val = not (len(seq) > self.max_seq_length)
        cont_val = not(('X' in seq) or ('B' in seq) or ('Z' in seq) or ('J' in seq))
        return len_val and cont_val

    # Generate a dictionary if none is supplied,
    def gen_acid_dict(self):
        for i, elem in enumerate(self.__acids__):
            temp = np.zeros(len(self.__acids__))
            temp[i] = 1
            self.acid_dict[elem] = temp
        return self.acid_dict

    # Read num_elems sequences from the given file
    def gen_data(self, num_elems):
        i = 0
        for record in self.__parser__:
            if (i == num_elems):
                break
            elif (not self.__is_legal_seq__(record.seq)):
                continue
            else:
                temp_full = np.full((self.max_seq_length - len(record), len(self.acid_dict)), self.acid_dict['-'])
                temp = np.array([self.acid_dict[elem] for elem in record])
                self.data.append(torch.tensor(np.concatenate((temp, temp_full), axis=0), dtype=torch.long))
                i += 1

'''
large_file = "uniref50.fasta"

data_gen = data_generator(large_file, 2000)
data_gen.gen_data(1000)
print(data_gen.acid_dict['A'])
print(data_gen.data[0])
print(data_gen.data[0].shape)
'''
