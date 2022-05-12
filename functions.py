# coding=utf-8

from pathlib import Path
from uuid import uuid4
from base64 import b64encode, b64decode
import os
from itertools import groupby, product
import nltk
from subprocess import PIPE, Popen
from lib.disambiguate import *
from lib.parameters import *


def find_nth(string, substring, n):
	start = string.find(substring)
	while start >= 0 and n > 1:
		start = string.find(substring, start + len(substring))
		n -= 1
	return start


def tr_lowercase(string):
	return string.replace('İ', 'i').replace('I', 'ı').lower()


def tr_uppercase(string):
	return string.replace('i', 'İ').replace('ı', 'I').upper()


def normalize_text(path, language):
	p = Path(path)
	print(p)
	outfile_path = str(p.parent) + '/c_' + p.name

	with open(path, 'r', encoding='utf-8') as infile, open(outfile_path, 'w', encoding='utf-8') as outfile:
		for line in infile:
			for replacement in replacements:
				line = line.replace(replacement, replacements[replacement])
			if language == 'en':
				line = line.lower()
			outfile.write(line)


def count_char_types(files: list):
	all_chars = {}

	for f in files:
		print(f)
		with open(f, 'r', encoding='utf-8') as infile:
			for line in infile:
				for char in line:
					if char not in all_chars:
						all_chars[char] = 1
					else:
						all_chars[char] += 1

	return {k: v for k, v in sorted(all_chars.items(), key=lambda item: item[1], reverse=True)}


def get_nones(files: list):
	nones = {}

	for file in files:
		print(file)
		with open(file, 'r', encoding='utf-8') as infile:
			infile.readline()
			for line in infile:
				tokens = tokenize(line)
				annotated_tokens = tag_tokens(tokens)

				for seq, annotated_token in enumerate(annotated_tokens):
					if annotated_token is None:
						if tokens[seq] not in nones:
							nones[tokens[seq]] = 1
						else:
							nones[tokens[seq]] += 1
	return nones


def tokenize(line: str, separate_punctuation: bool = False):
	if not separate_punctuation:
		return line.split(' ')
	else:
		count = 0
		for item in re.finditer(r'[^.?!:"\-,\']+', line):
			start_pos = item.span()[0]
			end_pos = item.span()[1]
			line = line[:start_pos + count] + ' ' + line[start_pos + count:]
			count += 1
			line = line[:end_pos + count] + ' ' + line[end_pos + count:]
			count += 1
		return line.strip().split()


def tag_tokens(tokens: list):
	token_types = [None] * len(tokens)
	for token_position, token in enumerate(tokens):
		for regex in regex_db:
			if re.match(regex, token) is not None:
				token_types[token_position] = regex_db[regex]
				break
	return token_types


def merge_apostrophes(annotated_tokens: list):
	for seq, token in enumerate(annotated_tokens):
		if token == 'AP' and seq + 1 < len(annotated_tokens):
			if annotated_tokens[seq - 1] in ('IU', 'AL', 'MC', 'IS', 'FP', 'WA', 'SN', 'AU', 'HW', 'IN', 'AB', 'RT') and \
					annotated_tokens[seq + 1] in ('AL', 'AU'):
				annotated_tokens[seq - 1] = 'WA'
				del annotated_tokens[seq]
				del annotated_tokens[seq]
	return annotated_tokens


def get_reduced_form(annotated_tokens: list):
	reduced_form = ''
	# Ignore certain tags
	# annotated_tokens = [i for i in annotated_tokens if i not in ('OP', 'CP')]
	# Categorize tags (beginning of sentence, word, mid-sentence punctuation, end of sentence, sentence boundary)
	for annotated_token in annotated_tokens:
		reduced_form += categories[annotated_token]
	return reduced_form


def sentence_validity(line_of_triple: str, remove_first_column: bool = False, separate_punctuation: bool = False):
	validity = False

	# Data
	if remove_first_column:
		linedata = line_of_triple.strip('\n').split('\t')[1:]
	else:
		linedata = line_of_triple.strip('\n').split('\t')
	left = tokenize(linedata[0], separate_punctuation)
	left_annotated = merge_apostrophes(tag_tokens(left))
	left_reduced = get_reduced_form(left_annotated)
	kwic = tokenize(linedata[1], separate_punctuation)
	kwic_annotated = merge_apostrophes(tag_tokens(kwic))
	kwic_reduced = get_reduced_form(kwic_annotated)
	right = tokenize(linedata[2], separate_punctuation)
	right_annotated = merge_apostrophes(tag_tokens(right))
	right_reduced = get_reduced_form(right_annotated)

	# Logic
	if (left_reduced[-3:] == 'WE|' or left_reduced[-2:] == 'WE') and \
			re.match(r'^B?W+(MW+)*E$', kwic_reduced) is not None and \
			kwic[-2:-1][0] not in abbreviations and \
			left[-3:-2][0] not in abbreviations and \
			kwic_annotated[-2:-1][0] != 'IS':
		if len(right_reduced) > 0:
			if (right_reduced[:2] == '|W' or right_reduced[0] == 'W'):
				validity = True
		else:
			validity = True
	return [validity, left, kwic, right, left_annotated, kwic_annotated, right_annotated, left_reduced, kwic_reduced,
			right_reduced]


def save_if_valid(line, path, remove_first_column: bool = False, separate_punctuation: bool = False):
	sentence_results = sentence_validity(line, remove_first_column, separate_punctuation)
	if sentence_results[0] is not False:
		sentence_length = str(sentence_results[8].count('W'))
		with open(path + str(sentence_length).rjust(3, '0') + '/' + str(uuid4()) + '.txt', 'w',
				  encoding='utf-8') as outfile:
			outfile.write(
				str(sentence_results[1]) + '\t' + str(sentence_results[2]) + '\t' + str(sentence_results[3]) + '\t' +
				str(sentence_results[4]) + '\t' + str(sentence_results[5]) + '\t' + str(sentence_results[6]) + '\t' +
				str(sentence_results[7]) + '\t' + str(sentence_results[8]) + '\t' + str(sentence_results[9]) + '\n')


def string_to_path(token: str):
	b64token = b64encode(token.encode('utf-8')).decode('utf-8').replace('/', '#')
	path = morph_store_root + str(ord(b64token[0]))
	if not os.path.isdir(path):
		os.makedirs(path)
	path += '/' + str(ord(b64token[1]))
	if not os.path.isdir(path):
		os.makedirs(path)
	path += '/' + str(ord(b64token[2]))
	if not os.path.isdir(path):
		os.makedirs(path)
	path += '/' + b64token[3:]
	return path


def path_to_string(path: str):
	pathdata = path.split('/')[-4:]
	b64token = chr(eval(pathdata[0])) + chr(eval(pathdata[1])) + chr(eval(pathdata[2])) + pathdata[3]
	return b64decode(b64token.replace('#', '/')).decode('utf-8')


def segment(token: str):
	p = Popen(segmentation_fst, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE)
	p.stdin.write(bytes(token, 'utf-8'))
	return {morph.replace('´', '\'').replace('ʼ', '\'').replace('’', '\'').replace('′', '\'').replace('-\'-', '-') for
			morph in p.communicate()[0].decode('utf-8').split('\n') if len(morph) > 0}


def generate_morphs(token: str, store: bool = True):
	p = Popen(analysis_fst, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE)
	p.stdin.write(bytes(token, 'utf-8'))
	morphs = p.communicate()[0].decode('utf-8').split('\n')
	# Use the following for Google morphological analyzer
	# morphs = analyze.surface_form(token, use_proper_feature=False)
	if store:
		path = string_to_path(tr_lowercase(token))
		try:
			with open(path, 'w', encoding='utf-8') as outfile:
				for morph in morphs:
					outfile.write(morph + '\n')
		except OSError:
			pass
	return list(filter(None, morphs))


def generate_token(morph):
	p = Popen(generation_fst, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE)
	p.stdin.write(bytes(morph, 'utf-8'))
	return p.communicate()[0].decode('utf-8').split('\n')[0].replace("'", "")


def eliminate_removable_analyses(morphs: list, remove_removables: bool = True):
	filtered_morphs = []
	for morph in morphs:
		acceptable = 1
		if morph != '':
			if remove_removables:
				for removable in removable_morphs:
					if removable in morph:
						acceptable = 0
						break

			if acceptable == 1 and morph not in filtered_morphs:
				filtered_morphs.append(morph)
	return filtered_morphs


def bundle_all_tags(morphs: list, merge_proper_names: bool = True):
	results = []
	for morph in morphs:
		tagdata = morph.split('<')
		stem = tagdata[0]
		tags = [tag_bundles['<' + tag] if '<' + tag in tag_bundles else '<' + tag for tag in tagdata[1:]]
		result = stem + ''.join(tags)
		if result not in results:
			results.append(result)

	if merge_proper_names:
		merged_results = []
		for result in results:
			if '<N:prop>' in result:
				if tr_lowercase(result[0]) + result[1:].replace('<N:prop>', '<N>') not in results and \
						tr_lowercase(result[0]) + result[1:].replace('<N:prop>', '<Advj>') not in results:
					merged_results.append(result)
			else:
				merged_results.append(result)
		return merged_results

	else:
		return results


def get_stem_pos_surface_tags(morphs: list):
	reduced_morphs = []
	for morph in morphs:
		try:
			stem = re.match(r'[^<]+', morph).group()
			tagdata = re.findall(r'<.+?>', morph)
			pos = tagdata[0]
			tagdata = tagdata[1:]

			# Remove all tags that do not have a surface representation
			suffixes = [x for x in tagdata if x not in non_surface_tags]

			# Remove consecutive duplicates
			suffixes = [x[0] for x in groupby(suffixes)]

			# Attach derivational suffixes to verb stem
			while len(suffixes) > 0 and suffixes[0] in vstem_tags:
				stem += '#'
				stem += suffixes[0].replace('<', '').replace('>', '')
				suffixes = suffixes[1:]
			stem = '<' + stem + '>'
			if (stem, pos, suffixes) not in reduced_morphs:
				reduced_morphs.append((stem, pos, suffixes))

		except AttributeError:
			return reduced_morphs

	return reduced_morphs


def merge_analyses(mlist):
	reduced_analyses = []
	for analysis in mlist:
		if analysis[1] == '<V>' and '#' in analysis[0]:
			generation_string = analysis[0][1:].replace('#', '<V><')
			reduced_analysis = ('<' + generate_token(generation_string) + '>', analysis[1], analysis[2])
			if reduced_analysis not in reduced_analyses:
				reduced_analyses.append(reduced_analysis)
		else:
			reduced_analyses.append(analysis)
	return reduced_analyses


def analyze(token: str, generate: bool = True, store: bool = True, mark_function_words: bool = True,
			remove_removables: bool = True, bundle_tags: bool = True, merge_proper_names: bool = True,
			triple_form: bool = True, probability_sorted: bool = True, merge_derivations: bool = True):
	if store and not generate:
		print('Cannot store without generating!')
		return None

	# Special tags for function words
	if mark_function_words:
		lowercase_token = tr_lowercase(token)
		if lowercase_token in function_words:
			if triple_form:
				return [('<' + lowercase_token + '>', '<' + lowercase_token + '>', [])]
			else:
				return [token + '<Func>']

	# Read analyses from file, generate and/or store if no file
	path = string_to_path(token)
	if os.path.isfile(path):
		with open(path, 'r', encoding='utf-8') as morphfile:
			morphs = morphfile.read().split('\n')
	else:
		if generate:
			if store:
				morphs = generate_morphs(token, True)
			else:
				morphs = generate_morphs(token, False)
		else:
			morphs = None

	# Remove empty analyses
	morphs = list(filter(None, morphs))

	# Mark out-of-vocabulary tokens
	if morphs[0] == '+?':
		if triple_form:
			return [('<' + token + '_Unk>', '<Unk>', [])]
		else:
			return [token + '<Unk>']

	# Sort analyses by probability (using disambiguate.py)
	if probability_sorted:
		morphs = disambiguate(morphs)

	if remove_removables:
		morphs = eliminate_removable_analyses(morphs, remove_removables=remove_removables)

	if bundle_tags:
		morphs = bundle_all_tags(morphs, merge_proper_names=merge_proper_names)

	if triple_form:
		morphs = get_stem_pos_surface_tags(morphs)

	if merge_derivations:
		morphs = merge_analyses(morphs)

	return morphs


def tag_sequences(sentence: str, stem_or_pos: str, best_n: int = 2, mark_function_words: bool = True,
				  remove_removables: bool = True,
				  bundle_tags: bool = True, merge_proper_names: bool = True, probability_sorted: bool = True,
				  word_boundaries: bool = True,
				  merge_derivations: bool = True):
	lattice = []
	results = []
	for token in list(filter(None, sentence.strip().split(' '))):
		analyses = analyze(token, mark_function_words=mark_function_words, remove_removables=remove_removables,
						   bundle_tags=bundle_tags, merge_proper_names=merge_proper_names, triple_form=True,
						   probability_sorted=probability_sorted, merge_derivations=merge_derivations)[:best_n]
		if stem_or_pos == 'pos':
			if word_boundaries:
				lattice.append([''.join([analysis[1]] + analysis[2] + ['|']) for analysis in analyses])
			else:
				lattice.append([''.join([analysis[1]] + analysis[2]) for analysis in analyses])
		elif stem_or_pos == 'stem':
			if word_boundaries:
				lattice.append([''.join([analysis[0]] + analysis[2] + ['|']) for analysis in analyses])
			else:
				lattice.append([''.join([analysis[0]] + analysis[2]) for analysis in analyses])
		else:
			print('\'stem_or_pos\' must be \'stem\' or \'pos\'')
	for item in list(product(*lattice)):
		# result = ''.join(item).replace('>|', '|>')
		result = ''.join(item)
		if result not in results:
			results.append(result)
	return results


def tag_ngrams(sentence: str, stem_or_pos: str, best_n: int = 2, ngram_length: int = 2,
			   mark_function_words: bool = True,
			   remove_removables: bool = True, bundle_tags: bool = True, merge_proper_names: bool = True,
			   probability_sorted: bool = True, merge_derivations: bool = True):
	results = []
	tag_seqs = tag_sequences(sentence, stem_or_pos, best_n, mark_function_words=mark_function_words,
							 remove_removables=remove_removables, bundle_tags=bundle_tags,
							 merge_proper_names=merge_proper_names,
							 probability_sorted=probability_sorted, merge_derivations=merge_derivations)
	for tag_seq in tag_seqs:
		for item in nltk.ngrams(re.findall(r'[^<>]+', tag_seq), ngram_length):
			results.append(item)
	return {i: results.count(i) for i in results}
