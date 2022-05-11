# coding=utf-8

from lib.functions import normalize_text, tag_sequences
from lib.parameters import *
from glob import glob
from random import sample
import re

# Step 1. Normalize text files
# (different versions of apostrophes, quotes, dashes; 'corrupted' Turkish characters, etc.; also convert to lowercase if English)
"""
csv_files = glob(csv_path + '/*.csv')
for csv_file in csv_files:
	normalize_text(csv_file, language)
"""

# Step 2. Merge CSV files to avoid repetitions caused by shuffling
"""
ngrams = {}

sketch_engine_ids = set()
csv_files = glob(csv_path + '/*.csv')

with open(csv_path + '/' + focus_token + '_merged_' + str(number_of_10k_results) + '.csv', 'w', encoding='utf-8') as outfile:
	for csv_file in csv_files:
		print(csv_file)
		with open(csv_file, 'r', encoding='utf-8') as csv_file:
			# Skip SketchEngine meta-information in top 5 lines
			for i in range(5):
				next(csv_file)
			for line in csv_file:
				sketch_engine_id = line.strip().split(",")[0][1:-1]
				if sketch_engine_id not in sketch_engine_ids:
					sketch_engine_ids.add(sketch_engine_id)
					outfile.write(line)
"""

# Step 3. Prepare ngrams
"""
with open(csv_path + '/' + focus_token + '_merged_' + str(number_of_10k_results) + '.csv', 'r', encoding='utf-8') as csvfile,\
 	open(ngrams_path + '_ngrams_' + str(number_of_10k_results) + '.tsv', 'w', encoding='utf-8') as ngram_file:
	for line_count, line in enumerate(csvfile):
		# Progress counter
		if line_count % 1000 == 0:
			print(line_count)
		line = line.replace(" ' ", "'")
		# Get SketchEngine ID, left context and right context
		sentence_data = line.strip().split('","')
		sketchengine_id = sentence_data[0][1:]
		sentence_data[1] = sentence_data[1].replace('""', '"')
		sentence_data[3] = sentence_data[3].replace('""', '"')
		# Prepare certain characters in English text files for tokenization (add necessary spaces before and after)
		if language == 'en':
			for english_separable in english_separables:
				sentence_data[1] = sentence_data[1].replace(english_separable, ' ' + english_separable + ' ')\
					.replace(' "', ' ').replace('" ', ' ').replace('"', '\'').strip()
				sentence_data[3] = sentence_data[3].replace(english_separable, ' ' + english_separable + ' ')\
					.replace(' "', ' ').replace('" ', ' ').replace('"', '\'').strip()
		# Re-combine processed tokens into a single piece
		sentence_fragment = re.split('\s+', sentence_data[1])[-left_window_size:] + [sentence_data[2]] + \
			re.split('\s+', sentence_data[3])[:right_window_size]
		# Mask any token that would 'break' (i.e. cannot occur in the middle of) a MWE
		modified_sentence_fragment = []
		for token in sentence_fragment:
			if token in breaking_tokens:
				modified_sentence_fragment.append('///')
			else:
				modified_sentence_fragment.append(token)
		# Morphologically analyze Turkish tokens
		if language == 'tr':
			sentence_fragment = ' '.join(modified_sentence_fragment).strip()
			# Experiment parameter
			# We have several options here (see functions.py for details)
			m_sequences = tag_sequences(sentence_fragment.strip(), 'stem', merge_derivations=False)
			sequences = []
			# Experiment parameter
			# If morphological ambiguity is too high, consider only n randomly selected analyses
			# This can be reduced, but not increased, in a post-processing step
			if len(m_sequences) > n_random_analyses:
				m_sequences = sample(m_sequences, n_random_analyses)
			for m_sequence in m_sequences:
				# Special treatment for the 2nd person singular imperative, considering it has no surface expression
				m_sequence = m_sequence.replace('><imp><2s>', '#imp2s>')
				m_sequence_data = m_sequence.split('|')[:-1]
				# Check if focus token is in the middle; deal with out-of-vocabulary items (<///_Unk>)
				if m_sequence_data[left_window_size].startswith('<' + focus_token + '>'):
					words = [item.split('><') for item in m_sequence_data]
					left_morphs = [morph.replace('<///_Unk>', '///') for word in words[:left_window_size][::-1]\
					for morph in word[::-1]][:left_window_size][::-1]
					right_morphs = [morph.replace('<///_Unk>', '///') for word in words[left_window_size:]\
					for morph in word][1:][:right_window_size]
					sequences.append(left_morphs + [focus_token] + right_morphs)
		elif language == 'en':
			sequences = [modified_sentence_fragment]
		else:
			print('Language not supported!')
			break
		# Spread ///s to the left and right, i.e. if L2 is ///, L3, L4 and L5 also must be ///
		for seq_id, sequence in enumerate(sequences):
			full_id = sketchengine_id + '_' + str(seq_id)
			try:
				left_idx = abs(sequence[:left_window_size][::-1].index('///') - left_window_size) - 2
			except ValueError:
				left_idx = None
			try:
				right_idx = sequence[left_window_size + 1:].index('///') + left_window_size + 2
			except ValueError:
				right_idx = None
			if left_idx is not None:
				for n in range(0, left_idx + 1):
					sequence[n] = '///'
			if right_idx is not None:
				for n in range(right_idx, len(sequence)):
					sequence[n] = '///'
			# Check if ngram is well-formed
			if len(sequence) == left_window_size + right_window_size + 1 and sequence[5].replace('<', '')\
				.replace('>', '') == focus_token:
				# Add prefixes (L1, KW, R1, etc.)
				ngram = [x[0] + '_' + x[1] for x in list(zip(prefixes, sequence))]
				# Write to file
				ngram_file.write(full_id + '\t' + str(ngram) + '\n')
"""

