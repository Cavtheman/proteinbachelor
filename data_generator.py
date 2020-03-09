from Bio import SeqIO
from string import ascii_uppercase


class data_generator():
    def __init__(self, filename, max_seq_length, acid_dict={}):
        if (acid_dict = {}):
            self.acid_dict = self.gen_acid_dict()
        else:
            self.acid_dict = acid_dict
        self.data = []
        self.max_seq_length = max_seq_length
        self.__parser__ = SeqIO.parse(filename, "fasta")
        self.__acids__ = "ACDEFGHIKLMNPQRSTVWY-"

    # If a sequence contains one of XBZJ it will be discarded,
    # and if it is longer than the given max_seq_length
    def __is_legal_seq__(self, seq):
        len_val = not (len(seq) > self.max_seq_length)
        cont_val = not(('X' in seq) or ('B' in seq) or ('Z' in seq) or ('J' in seq))
        return len_val and cont_val

    # Generate a dictionary if none is supplied\n",
    def gen_acid_dict(self):
        for i, elem in enumerate(self.__acids__):
            temp = np.zeros(len(self.__acids__))
            temp[i] = 1
            self.acid_dict[elem] = temp
        return self.acid_dict

    # Read num_elems sequences from the given file
    def gen_data(self, num_elems):
        for i, record in enumerate(self.__parser__):
            # [self.acid_dict[aa] for aa in str(record.seq).upper()]
            if (i == num_elems):
                break
            elif (not self.__is_legal_seq__(record.seq)):
                continue
            else:
                self.data.append(str(record.seq).upper())
