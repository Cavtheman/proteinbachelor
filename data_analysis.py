import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils import data
from data_generator import Dataset
from print_seq import print_seq
from collections import Counter


acids = "ACDEFGHIKLMNOPQRSTUVWY-"
large_file = "uniref50.fasta"
small_file = "100k_rows.fasta"

#max_seq_len = float("inf")
max_seq_len = 2000
batch_size = 32

dataset = Dataset(small_file, max_seq_len, acids=acids)

#base_generator = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# autolabel taken from https://matplotlib.org/examples/api/barchart_demo.html
def autolabel(rects, index):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax[index].text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                       ha='center', va='bottom', rotation=45)

letter_counts = {}
seq_lens = []

#'''
for seq in dataset.data:
    temp = Counter(seq)
    seq_lens.append(len(seq))
    for count in temp:
        if count in letter_counts:
            letter_counts[count] += temp[count]
        else:
            letter_counts[count] = temp[count]
#'''

#letter_counts = Counter(dataset.data[0])

sorted_data = list(letter_counts.items())
sorted_data.sort()

sorted_keys1 = [elem[0] for elem in sorted_data]
sorted_vals1 = [elem[1] for elem in sorted_data]

sorted_data.sort(key=lambda tup: tup[1], reverse=True)

sorted_keys2 = [elem[0] for elem in sorted_data]
sorted_vals2 = [elem[1] for elem in sorted_data]


freq_fig, freq_ax = plt.subplots(2, figsize=(12, 8))

num_vals = len(letter_counts.values())

bars1 = freq_ax[0].bar(np.linspace(0,num_vals,num_vals), sorted_vals1, 1)
freq_ax[0].set_xticks(np.linspace(0,num_vals,num_vals))
freq_ax[0].set_xticklabels(sorted_keys1)
#freq_ax[0].set_ylabel('right / top')


bars2 = freq_ax[1].bar(np.linspace(0,num_vals,num_vals), sorted_vals2, 1)
freq_ax[1].set_xticks(np.linspace(0,num_vals,num_vals))
freq_ax[1].set_xticklabels(sorted_keys2)
#freq_ax[1].set_ylabel('left / bottom')

#autolabel(bars1, 0)
#autolabel(bars2, 1)
freq_fig.savefig("char_frequency.png")
plt.show()


len_fig, len_ax = plt.subplots(1, figsize=(12, 8))

len_ax.hist(seq_lens, bins='auto')
len_fig.savefig("len_hist.png")
plt.show()
