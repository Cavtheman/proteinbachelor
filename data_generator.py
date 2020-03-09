from Bio import SeqIO
from string import ascii_uppercase

class data_generator():
    def __init__(self, filename, max_seq_length, acid_dict={}):
        if (acid_dict == {}):
            self.acid_dict = self.gen_acid_dict()
        else:
            self.acid_dict = acid_dict
        self.data = []
        self.max_seq_length = max_seq_length
        self.__parser__ = SeqIO.parse(filename, "fasta")

    # If a sequence contains one of XBZJ it will be discarded,
    # and if it is longer than the given max_seq_length
    def __is_legal_seq__(self, seq):
        len_val = not (len(seq) > self.max_seq_length)
        cont_val = not(('X' in seq) or ('B' in seq) or ('Z' in seq) or ('J' in seq))
        return len_val and cont_val

    # Generate a dictionary if none is supplied\n",
    def gen_acid_dict(self):
        num = 0
        acid_dict = {}
        for letter in ascii_uppercase:
            acid_dict[str(letter)] = num
            num += 1
        acid_dict["NOSEQ"] = num
        self.acid_dict = acid_dict
        return acid_dict

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
