# coding=utf-8

from lib.parameters import *
import numpy as np
from collections import defaultdict
from math import log2
from operator import itemgetter
from os.path import isfile
from itertools import product


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


subgram_freqs = defaultdict(int, default_factory=None)
token_freqs = defaultdict(int, default_factory=None)
position_counts = defaultdict(int, default_factory=0)

print('Loading subgram frequencies...')
with open(subgram_path, 'r', encoding='utf-8') as subgram_file:
	for line_count, line in enumerate(subgram_file):
		if line_count % 100000 == 0:
			print(line_count)
		linedata = line.strip().split('\t')
		subgram = eval(linedata[0])
		subgram_freq = eval(linedata[1])
		subgram_freqs[subgram] = subgram_freq

print('Loading token frequencies...')
with open(token_path, 'r', encoding='utf-8') as token_file:
	for line in token_file:
		linedata = line.strip().split('\t')
		token = linedata[0]
		position_counts[token.split('_')[0]] += 1
		token_freq = eval(linedata[1])
		token_freqs[token] = token_freq

print('Loading aggregate matrix...')
aggregate_matrix = np.load(aggregate_matrix_path)
normalized_aggregate_matrix = aggregate_matrix / np.sum(aggregate_matrix)

"""
deduct_matrices_options = [True, False]
trim_matrices_options = [True]
expected_frequency_adjustment_options = [2, 4, 8]
score_calculation_options = ['expected', 'aggregate']
score_adjustment_options = ['none', 'divide-by-length']
number_of_winners_options = [1, 2, 3]
min_score_options = [0, 0.50, 1, 2]
combine_winners_options = ['add-score', 'add-one', 'max']
"""

deduct_matrices_options = [True]
trim_matrices_options = [True]
expected_frequency_adjustment_options = [2]
score_calculation_options = ['expected']
score_adjustment_options = ['none']
number_of_winners_options = [1]
min_score_options = [0]
combine_winners_options = ['add-one']

all_parameters = [deduct_matrices_options, trim_matrices_options, expected_frequency_adjustment_options, score_calculation_options, score_adjustment_options, number_of_winners_options, min_score_options, combine_winners_options]
parameter_combinations = list(product(*all_parameters))

for parameter_combination in parameter_combinations:
	deduct_matrices = parameter_combination[0]
	trim_matrices = parameter_combination[1]
	expected_frequency_adjustment = parameter_combination[2]
	score_calculation = parameter_combination[3]
	score_adjustment = parameter_combination[4]
	number_of_winners = parameter_combination[5]
	min_score = parameter_combination[6]
	combine_winners = parameter_combination[7]
	filepath = results_path + focus_token + '_' + '_'.join([str(x) for x in parameter_combination]) + '.tsv'
	combined_winners = defaultdict(int)

	if isfile(filepath):
		print('Focus token: ' + focus_token + ' | Already processed: ' + str(parameter_combination))
	else:
		print('Focus token: ' + focus_token + ' | Now processing: ' + str(parameter_combination))
		with open('data/' + language + '/01_ngrams/' + focus_token + '/' + focus_token + '_ngrams_' + str(number_of_10k_results) + '.tsv', 'r', encoding='utf-8') as infile:
			for line_count, line in enumerate(infile):
				if line_count % 10000 == 0:
					print(line_count)
				linedata = line.strip().split('\t')
				focus_ngram_id = linedata[0]
				focus_ngram = eval(linedata[1])
				focus_token_freqs = [token_freqs[token] for token in focus_ngram]
				total_freq = focus_token_freqs[left_window_size]
				left_probs_pos = np.array(focus_token_freqs[:left_window_size][::-1]) / total_freq
				right_probs_pos = np.array(focus_token_freqs[left_window_size + 1:]) / total_freq
				left_probs_neg = np.append((1 - left_probs_pos), 1)
				right_probs_neg = np.append((1 - right_probs_pos), 1)
				coselection_matrix = np.zeros((left_window_size + 1, right_window_size + 1), dtype=int)
				matrix_winners = []

				# generate initial version of coselection matrix
				for left_idx in left_indices:
					for right_idx in right_indices:
						seq = [focus_ngram[item] for item in left_idx + right_idx]
						if tuple(seq) in subgram_freqs:
							freq = subgram_freqs[tuple(seq)]
						else:
							freq = 0
						coselection_matrix[len(left_idx) - 1, len(right_idx)] = freq

				if deduct_matrices:
					coselection_matrix = deduct_repetitions(coselection_matrix, left_window_size, right_window_size)

				expected_matrix = np.zeros((left_window_size + 1, right_window_size + 1), dtype=int)

				for left_idx in range(left_window_size + 1):
					for right_idx in range(right_window_size + 1):
						probs = np.concatenate([left_probs_pos[:left_idx], [left_probs_neg[left_idx]], right_probs_pos[:right_idx], [right_probs_neg[right_idx]]])
						expected_matrix[left_idx, right_idx] = np.prod(probs) * total_freq

				if trim_matrices:
					mask = [0 if token[-3:] == '///' else 1 for token in focus_ngram]
					row_limit = sum(mask[:left_window_size]) + 1
					column_limit = sum(mask[left_window_size + 1:]) + 1
					current_coselection_matrix = coselection_matrix[:row_limit, :column_limit]
					current_expected_matrix = expected_matrix[:row_limit, :column_limit]
					current_aggregate_matrix = normalized_aggregate_matrix[:row_limit, :column_limit]
					current_length_matrix = length_matrix[:row_limit, :column_limit]
				else:
					current_coselection_matrix = coselection_matrix
					current_expected_matrix = expected_matrix
					current_aggregate_matrix = normalized_aggregate_matrix
					current_length_matrix = length_matrix

				# calculate scores
				scores = np.zeros_like(current_coselection_matrix, dtype=float)
				if score_calculation == 'expected':
					for n in range(current_coselection_matrix.shape[0]):
						score_row = np.array([log2(x+1) for x in current_coselection_matrix[n]]) / np.array([log2(int(x) + expected_frequency_adjustment) for x in current_expected_matrix[n]])
						#score_row = np.log2(np.array([x + 1 for x in current_coselection_matrix[n]]) / np.array([int(x) + expected_frequency_adjustment for x in current_expected_matrix[n]]))
						scores[n] = score_row
				elif score_calculation == 'aggregate':
					current_coselection_matrix = current_coselection_matrix / total_freq
					scores = np.divide(current_coselection_matrix, current_aggregate_matrix)
				else:
					print('WARNING: Undefined score_calculation parameter!')

				if score_adjustment == 'divide-by-length':
					scores = scores / current_length_matrix

				# add n top-scoring sub-grams
				max_score_rows, max_score_columns = [], []

				for n in reversed(range(0, number_of_winners + 1)):
					try:
						max_score_rows, max_score_columns = np.unravel_index(np.argpartition(scores.ravel(), len(scores.ravel()) - n)[-n:], scores.shape)
						break
					except ValueError:
						pass

				for idx in list(zip(max_score_rows, max_score_columns)):
					max_sequence = focus_ngram[left_window_size-idx[0]: left_window_size + idx[1] + 1]
					if language == 'tr':
						if '<' not in ' '.join([x for x in max_sequence]) and '>' not in ' '.join([x for x in max_sequence[:-1]]):
							pass
						else:
							hit = ' '.join([x.split('_')[1] for x in max_sequence]).replace(' ///', '').replace('/// ', '')
							score = scores[idx[0], idx[1]]
							if score >= min_score:
								matrix_winners.append((hit, score))
					elif language == 'en':
						hit = ' '.join([x.split('_')[1] for x in max_sequence]).replace(' ///', '').replace('/// ', '')
						score = scores[idx[0], idx[1]]
						if score >= min_score:
							matrix_winners.append((hit, score))

				# Combine individual matrix winners using various methods
				for matrix_winner in matrix_winners:
					h = matrix_winner[0]
					s = matrix_winner[1]
					if combine_winners == 'add-score':
						combined_winners[h] += s
					elif combine_winners == 'add-one':
						combined_winners[h] += 1
					elif combine_winners == 'max':
						if combined_winners[h] < s:
							combined_winners[h] = s
					else:
						print('WARNING: Undefined combined_winner parameter!')

			# Sort global results
			sorted_combined_winners = sorted(combined_winners.items(), key=itemgetter(1), reverse=True)

			# Write to file
			with open(filepath, 'w', encoding='utf-8') as result_file:
				for item in sorted_combined_winners[:evaluation_size * 10]:
					result_file.write(item[0] + '\t' + str(item[1]) + '\n')

