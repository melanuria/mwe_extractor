# coding=utf-8
import io

from lib.parameters import *
import matplotlib.pyplot as plt
from glob import glob
from collections import defaultdict
from operator import itemgetter
from itertools import combinations


# generate naive ngrams as a baseline
"""
naive_ngrams = defaultdict(int)
with open(ngrams_path + focus_token + '_ngrams_' + str(number_of_10k_results) + '.tsv', 'r', encoding='utf-8') as ngram_file:
	for line_count, line in enumerate(ngram_file):
		linedata = line.strip().split('\t')
		ngram = eval(linedata[1])
		processed_seqs = []
		for left_idx in left_indices:
			for right_idx in right_indices:
				seq = [ngram[item] for item in left_idx + right_idx if '///' not in ngram[item]]
				if seq not in processed_seqs:
					processed_seqs.append(seq)
					naive_ngrams[tuple(seq)] += 1

sorted_naive_ngrams = sorted(naive_ngrams.items(), key=itemgetter(1), reverse=True)

with open(ngrams_path + focus_token + '_naive.tsv', 'w', encoding='utf-8') as naive_file:
	for count, item in enumerate(sorted_naive_ngrams):
		if count <= evaluation_size * 10:
			subgram = ' '.join([x.split('_')[1] for x in item[0]])
			if language == 'tr':
				if ('<' not in subgram and '>' not in subgram[:-1]) is False:
					naive_file.write(subgram + '\t' + str(item[1]) + '\n')
			elif language == 'en':
				naive_file.write(subgram + '\t' + str(item[1]) + '\n')
			else:
				print('Language not supported...')
"""

# get union of all result files in a folder
"""
gold_mwes = []

with open(ngrams_path + focus_token + '_gold.txt', 'r', encoding='utf-8') as gold_file:
	try:
		for line in gold_file:
			gold_mwes.append(line.strip()[2:])
	except io.UnsupportedOperation:
		pass

files = glob(results_path + focus_token + '*.tsv')
# Add 'naive' file, which is in another folder
files.append(ngrams_path + focus_token + '_naive.tsv')

mwes = []
for file in files:
	with open(file, 'r', encoding='utf-8') as infile:
		candidates = []
		for x in range(evaluation_size):
			try:
				candidates.append(next(infile))
			except StopIteration:
				break
		for line in candidates:
			linedata = line.strip().split('\t')
			if linedata[0] not in mwes:
				# exclude single words
				if language == 'tr':
					if ('<' not in linedata[0] and '>' not in linedata[0][:-1]) is False:
						mwes.append(linedata[0])
				elif language == 'en':
					mwes.append(linedata[0])
				else:
					print('Language not supported...')

for mwe in mwes:
	if mwe not in gold_mwes:
		print('*_' + mwe)
"""


# add new unique items to gold file
"""
existing = set()

with open(ngrams_path + focus_token + '_gold.txt', 'r', encoding='utf-8') as gold_file,\
		open(ngrams_path + focus_token + '_additions.txt', 'r', encoding='utf-8') as add_file,\
		open(ngrams_path + focus_token + '_approved_additions.txt', 'w', encoding='utf-8') as approved_file:
	for line1 in gold_file:
		existing.add(line1.strip().split('_', 1)[1])
	for line2 in add_file:
		if line2.strip().split('_', 1)[1] not in existing:
			print('New line: ' + line2.strip())
			approved_file.write(line2)
"""


# calculate precision of each result file in a folder
"""
valid_mwes = []

files = glob(results_path + '*.tsv')
files.append(ngrams_path + focus_token + '_naive.tsv')

with open(ngrams_path + focus_token + '_gold.txt', 'r', encoding='utf-8') as gold_file:
	for line in gold_file:
		if line[:2] == '1_':
			valid_mwes.append(line.strip()[2:])

print('Total gold: ' + str(len(valid_mwes)))

for file in files:
	valid_count = 0
	total_count = 0
	with open(file, 'r', encoding='utf-8') as infile:
		candidates = []
		for x in range(evaluation_size * 2):
			try:
				cand = next(infile)
				# Ignore single words
				if language == 'tr':
					if ('<' not in cand and '>' not in cand[:-1]) is False:
						candidates.append(cand)
				elif language == 'en':
					if ' ' in cand:
						candidates.append(cand)
			except StopIteration:
				break

		for line in candidates[:evaluation_size]:
			total_count += 1
			candidate = line.strip().split('\t')[0]
			if candidate in valid_mwes:
				valid_count += 1
	try:
		print('\t'.join([str(x) for x in file.split('/')[4].split('_')[1:]]) + '\t' + str(valid_count) + '\t' + str(total_count))
	except ZeroDivisionError:
		print('Empty file: ' + '\t'.join([str(x) for x in file.split('/')[5].split('_')[1:]]))
"""


# calculate combined performance of all two-method pairs
"""
valid_mwes = []

with open(ngrams_path + focus_token + '_gold.txt', 'r', encoding='utf-8') as gold_file:
	for line in gold_file:
		if line[:2] == '1_':
			valid_mwes.append(line.strip()[2:])

files = glob(results_path + '*.tsv')
files.append(ngrams_path + focus_token + '_naive.tsv')

file_pairs = combinations(files, 2)

combination_results = {}

for pair_count, file_pair in enumerate(file_pairs):
	true_positive_count = 0
	candidates = []
	if pair_count % 10000 == 0:
		print(pair_count)
	for file in file_pair:
		with open(file, 'r', encoding='utf-8') as infile:
			for line_count, line in enumerate(infile):
				if line_count < int(evaluation_size / 2):
					candidate = line.strip().split('\t')[0]
					# Ignore single words
					if language == 'tr':
						if ('<' not in candidate and '>' not in candidate[:-1]) is False and candidate not in candidates:
							candidates.append(candidate)
					elif language == 'en':
						if ' ' in candidate and candidate not in candidates:
							candidates.append(candidate)
				else:
					break

	for candidate in candidates:
		if candidate in valid_mwes:
			true_positive_count += 1

	if len(candidates) > 0:
		combination_results[file_pair[0], file_pair[1]] = (true_positive_count, len(candidates), true_positive_count / len(candidates))
	else:
		combination_results[file_pair[0], file_pair[1]] = (0, 0, 0)

sorted_combination_results = sorted(combination_results.items(), key=lambda x: x[1][2])

with open('x.txt', 'w', encoding='utf-8') as outfile:
	for item in sorted_combination_results:
		outfile.write(str(item[0]) + '\t' + str(item[1][0]) + '\t' + str(item[1][1]) + '\t' + str(item[1][2]) + '\n')
"""

# combine the results of two methods
"""
file1 = results_path + focus_token + '_True_True_4_expected_divide-by-length_3_0.5_max.tsv'
file2 = results_path + focus_token + '_True_True_4_aggregate_none_3_2_add-one.tsv'
results1, results2 = [], []

with open(file1, 'r', encoding='utf-8') as infile1,\
		open(file2, 'r', encoding='utf-8') as infile2:
	for line in infile1:
		results1.append(line.strip().split('\t')[0])
	for line in infile2:
		results2.append(line.strip().split('\t')[0])

if len(results1) >= len(results2):
	long_list = results1
	short_list = results2
else:
	long_list = results2
	short_list = results1


insertion_point = 1

for item in short_list:
	if item not in long_list:
		long_list.insert(insertion_point, item)
		insertion_point += 2

with open(results_path + 'combined_methods/' + '_'.join(file1.split('_')[3:])[:-4] + '#' + '_'.join(file2.split('_')[3:]), 'w', encoding='utf-8') as outfile:
	for item in long_list:
		outfile.write(item + '\n')
"""


# plot the precision profile of one or more methods

font = {'size': 34}
plt.rc('font', **font)

valid_mwes = []

with open(ngrams_path + focus_token + '_gold.txt', 'r', encoding='utf-8') as gold_file:
	for line in gold_file:
		if line[:2] == '1_':
			valid_mwes.append(line.strip()[2:])

files = glob(results_path + focus_token + '*.tsv')
files.append(ngrams_path + focus_token + '_naive.tsv')

for file_sequence, file in enumerate(files):
	file_props = file.split('/')[4].split('_')[1:]
	if len(file_props) < 8:
		file_props = [0, 0, 0, 0, 0, 0, 0, 0]
	precisions = []
	sum_of_precisions = 0
	true_positive_count = 0
	with open(file, 'r') as infile:
		for rank, line in enumerate(infile):
			if rank < evaluation_size:
				candidate = line.strip().split('\t')[0]
				if candidate in valid_mwes:
					true_positive_count += 1
					sum_of_precisions += true_positive_count / (rank + 1)
				precisions.append(true_positive_count / (rank + 1))
			else:
				break

		x = range(1, len(precisions) + 1)

		if (
				file_props[0] in ['True'] and
				file_props[1] in ['True'] and
				file_props[2] in ['2'] and
				file_props[3] in ['expected'] and
				file_props[4] in ['none'] and
				file_props[5] in ['1'] and
				file_props[6] in ['0'] and
				file_props[7] in ['add-one.tsv']		):
			col = 'black'
			lw = 5
			ls = 'solid'
			z = 3

		else:
			col = 'gray'
			lw = 0.5
			ls = 'solid'
			z = 1
		if file_props[0] == 0:
			col = 'black'
			lw = 5
			ls = 'dashed'
			z = 2
			print(precisions)

		plt.plot(x, precisions, linestyle=ls, linewidth=lw, color=col, zorder=z, marker=None)

plt.xlabel('Top n candidates ($\it{time}$)', labelpad=5)
plt.ylabel('Precision', labelpad=5)

plt.show()
