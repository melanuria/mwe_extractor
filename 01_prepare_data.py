# coding=utf-8

from lib.functions import *
from collections import defaultdict
import operator
import numpy as np


def deduct_repetitions(matrix, left_window, right_window):
	flipped_matrix = np.flip(matrix)
	diagonals = {k: [] for k in range(left_window + right_window + 1)}
	for (row, column), value in np.ndenumerate(flipped_matrix):
		diagonals[row + column].append((row, column))
	for diag_no, elements in diagonals.items():
		for element in elements:
			subtraction_matrix = np.zeros((left_window + 1, right_window + 1), dtype=int)
			subtractable_row = element[0]
			subtractable_column = element[1]
			subtractable_freq = flipped_matrix[subtractable_row][subtractable_column]
			if subtractable_freq > 0:
				for (row, column), freq in np.ndenumerate(flipped_matrix):
					if row >= subtractable_row and column >= subtractable_column and (row, column) != (
					subtractable_row, subtractable_column):
						subtraction_matrix[row][column] += subtractable_freq
				flipped_matrix = flipped_matrix - subtraction_matrix
	# convert negative values to zero and re-flip
	return np.flip(np.where(flipped_matrix < 0, 0, flipped_matrix))

subgram_freqs = defaultdict(int, default_factory=0)
token_freqs = defaultdict(int, default_factory=0)
aggregate_matrix = np.zeros((left_window_size + 1, right_window_size + 1), dtype=int)

with open(ngrams_path + focus_token + '_ngrams_' + str(number_of_10k_results) + '.tsv', 'r', encoding='utf-8') as infile:
	print('Calculating token frequencies and subgram frequencies...')
	for line_count, line in enumerate(infile):
		if line_count % 100000 == 0:
			print(line_count)
		ngram = eval(line.strip().split('\t')[1])
		for token in ngram:
			token_freqs[token] += 1

		processed_seqs = []
		for left_idx in left_indices:
			for right_idx in right_indices:
				seq = [ngram[item] for item in left_idx + right_idx if '///' not in ngram[item]]
				if seq not in processed_seqs:
					processed_seqs.append(seq)
					subgram_freqs[tuple(seq)] += 1

	# return to start of ngram file and process again for aggregate matrix
	infile.seek(0)

	print('Calculating aggregate matrix...')
	for line_count, line in enumerate(infile):
		if line_count % 100000 == 0:
			print(line_count)
		linedata = line.strip().split('\t')
		focus_ngram = eval(linedata[1])
		coselection_matrix = np.zeros((left_window_size + 1, right_window_size + 1), dtype=int)
		for left_idx in left_indices:
			for right_idx in right_indices:
				seq = [focus_ngram[item] for item in left_idx + right_idx]
				if tuple(seq) in subgram_freqs:
					freq = subgram_freqs[tuple(seq)]
				else:
					freq = 0
				coselection_matrix[len(left_idx) - 1, len(right_idx)] = freq
		coselection_matrix = deduct_repetitions(coselection_matrix, left_window_size, right_window_size)
		aggregate_matrix += coselection_matrix

sorted_subgram_freqs = sorted(subgram_freqs.items(), key=operator.itemgetter(1), reverse=True)
sorted_token_freqs = sorted(token_freqs.items(), key=operator.itemgetter(1), reverse=True)

with open(ngrams_path + focus_token + '_subgram_freqs_min_' + str(min_freq) + '_' + str(number_of_10k_results) + '.tsv', 'w', encoding='utf-8') as subgram_file:
	for item in sorted_subgram_freqs:
		if item[1] >= min_freq:
			subgram_file.write(str(item[0]) + '\t' + str(item[1]) + '\n')

with open(ngrams_path + focus_token + '_token_freqs_min_' + str(min_freq) + '_' + str(number_of_10k_results) + '.tsv', 'w', encoding='utf-8') as token_file:
	for item in sorted_token_freqs:
		if item[1] >= min_freq:
			token_file.write(str(item[0]) + '\t' + str(item[1]) + '\n')

with open(aggregate_matrix_path, 'wb') as aggregate_file:
	np.save(aggregate_matrix_path, aggregate_matrix)
