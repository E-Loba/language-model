import pickle
from random import shuffle, sample, random
import nltk
from gzip import open as zopen
# from taggers import ConsistentRandomTagger
from taggers import BayesTagger, StackTagger, feature_detector
from z_util import *


def make_cola():
	"""
	write formats:
	lex input: target -> series of contexts
	pos input: target -> series of contexts
	targets file: turns -> lex target -> pos target -> label targets
	:return:
	"""
	tagger_name = "pickle_jar/brown_full_tagger.pickle"
	with open(tagger_name, "rb") as dict_file:
		tagger = pickle.load(dict_file)
	with open(pos_tok_name, "rb") as dict_file:
		pos_tokeniser = pickle.load(dict_file)
	print("loaded POS tokeniser")
	with open(lex_tok_name, "rb") as dict_file:
		lex_tokeniser = pickle.load(dict_file)
	pos_start = [pos_tokeniser.num_words + 1]
	pos_end = [pos_tokeniser.num_words + 2]
	lex_start = [lex_tokeniser.num_words + 1]
	lex_end = [lex_tokeniser.num_words + 2]
	
	training = True
	if training:
		raw_file = "C:/Users/admin/Downloads/GLUE data/CoLA/train.tsv"
		write_file_lex = "cola_train_lex.gz"
		write_file_pos = "cola_train_pos.gz"
		write_file_tar = "cola_train_tar.gz"
	else:
		raw_file = "C:/Users/admin/Downloads/GLUE data/CoLA/dev.tsv"
		write_file_lex = "cola_dev_lex.gz"
		write_file_pos = "cola_dev_pos.gz"
		write_file_tar = "cola_dev_tar.gz"
	lines = []
	with open(raw_file, "r", encoding='utf-8') as read_file:
		for raw_line in read_file:
			line = raw_line.strip().split('\t')
			lines.append((line[1], line[3]))
	shuffle(lines)
	with zopen(write_file_lex, "wb") as w_f_l, zopen(write_file_pos, "wb") as w_f_p, zopen(write_file_tar, "wb") as w_f_t:
		for line in lines:
			label, string = line
			label = ' '.join([str(tkn) for tkn in make_one_hot(int(label), 3)])
			string = prepare_raw_string("determine acceptability: " + string)
			tags = [t for w, t in tagger.tag(string)]
			# print(string)
			# print(tags)
			string = lex_tokeniser.texts_to_sequences([string])[0]
			string = pad_list(lex_start + string + lex_end, seq_pad_len)
			string = ' '.join([str(tkn) for tkn in string])
			# print(string)
			w_f_l.write((string + '\n').encode(encoding=encoding))
			tags = pos_tokeniser.texts_to_sequences([tags])[0]
			tags = pad_list(pos_start + tags + pos_end, seq_pad_len)
			tags = ' '.join([str(tkn) for tkn in tags])
			# print(tags)
			w_f_p.write((tags + '\n').encode(encoding=encoding))
			write_ = delimitter.join(["0", string, tags, label])
			# print(write_)
			w_f_t.write((write_ + '\n').encode(encoding=encoding))
	return


def make_input_files(paragraphs_raw):
	"""
	[ paragraph 1 -> [ sent1=[(w1, t1), ...], sent2=[...], ... ],
	  paragraph 2 -> ...
	]
	:param paragraphs_raw: list of paragraphs
	:return:
	"""
	paragraphs = []
	for p in paragraphs_raw:
		paragraph_ = []
		for turn, sent in enumerate(p):
			paragraph_.extend([(turn % 2, list_sent) for list_sent in truncate_split(sent, max_sent_len, overlap)])
		paragraphs.append(paragraph_)
	del paragraphs_raw
	lengths = [len(p) for p in paragraphs]
	print("made paragraphs")
	
	paras_num = len(paragraphs)
	indices_in_nested_list = [(p, s) for p in range(paras_num) for s in range(1, lengths[p])]
	# first index, p = index of list in list. second index, s = index of item in nested list
	# second loop starts at 1 to skip the first (0) index in each nested list
	jumbled = [(p-paras_num, s) for p, s in indices_in_nested_list]
	indices_in_nested_list.extend(jumbled)
	shuffle(indices_in_nested_list)
	total = sum(lengths)
	print("total sentences: {}".format(total))
	print("made index list")

	with open(pos_tok_name, "rb") as dict_file:
		pos_tokeniser = pickle.load(dict_file)
	print("loaded POS tokeniser")
	with open(lex_tok_name, "rb") as dict_file:
		lex_tokeniser = pickle.load(dict_file)
	print("loaded lexical tokeniser")
	pos_start = [pos_tokeniser.num_words + 1]
	pos_end = [pos_tokeniser.num_words + 2]
	lex_start = [lex_tokeniser.num_words + 1]
	lex_end = [lex_tokeniser.num_words + 2]

	with zopen(data_name + '_lex_inputs.gz', 'wb') as in_lex_f, zopen(data_name + '_targets.gz', 'wb') as tar_f, zopen(data_name + '_pos_inputs.gz', 'wb') as in_pos_f:
		for index_1, index_2 in indices_in_nested_list:
			jumble = index_1 < 0
			turn, target = paragraphs[index_1][index_2]
			turns = [turn]
			if jumble:
				contexts = [sample(target, len(target))]
			else:
				contexts = [target]
			for index_3 in range(index_2):
				turn, context = paragraphs[index_1][index_3]
				turns.append(turn)
				contexts.append(context)
			tar_lex, tar_pos = zip(*target)
			tar_lex = lex_tokeniser.texts_to_sequences([list(tar_lex)])[0] + lex_end
			tar_lex = pad_list(tar_lex, seq_pad_len)
			tar_pos = pos_tokeniser.texts_to_sequences([list(tar_pos)])[0] + pos_end
			tar_pos = pad_list(tar_pos, seq_pad_len)
			write_ = delimitter.join([' '.join(str(turn) for turn in turns), ' '.join([str(lex_) for lex_ in tar_lex]),
									  ' '.join([str(pos_) for pos_ in tar_pos]), str(-1 if jumble else 1)])
			tar_f.write((write_ + '\n').encode(encoding=encoding))
			in_lex, in_pos = zip(*[tuple(zip(*sent)) for sent in contexts])
			in_lex = list(in_lex)
			in_pos = list(in_pos)
			for count in range(len(in_lex)):
				in_lex[count] = lex_tokeniser.texts_to_sequences([list(in_lex[count])])[0]
				in_pos[count] = pos_tokeniser.texts_to_sequences([list(in_pos[count])])[0]
				if count != 0:
					in_lex[count] += lex_end
					in_pos[count] += pos_end
				in_lex[count] = lex_start + in_lex[count]
				in_lex[count] = pad_list(in_lex[count], seq_pad_len)
				in_lex[count] = ' '.join(str(lex_) for lex_ in in_lex[count])
				
				in_pos[count] = pos_start + in_pos[count]
				in_pos[count] = pad_list(in_pos[count], seq_pad_len)
				in_pos[count] = ' '.join(str(pos_) for pos_ in in_pos[count])
			write_ = delimitter.join(in_lex)
			in_lex_f.write((write_ + '\n').encode(encoding=encoding))
			write_ = delimitter.join(in_pos)
			in_pos_f.write((write_ + '\n').encode(encoding=encoding))
	return


def make_mlm_inputs(sentences_raw):
	indices = list(range(len(sentences_raw)))
	shuffle(indices)
	
	with open(pos_tok_name, "rb") as dict_file:
		pos_tokeniser = pickle.load(dict_file)
	print("loaded POS tokeniser")
	with open(lex_tok_name, "rb") as dict_file:
		lex_tokeniser = pickle.load(dict_file)
	print("loaded lexical tokeniser")
	pos_start = pos_tokeniser.num_words + 1
	pos_end = pos_tokeniser.num_words + 2
	pos_mask = 1
	lex_start = lex_tokeniser.num_words + 1
	lex_end = lex_tokeniser.num_words + 2
	lex_mask = 1
	
	valid_inputs_count = 0
	with zopen(data_name + "_mlm_lex_inputs.gz", "wb") as lex_f, zopen(data_name + "_mlm_pos_inputs.gz", "wb") as pos_f, zopen(data_name + "_mlm_targets.gz", "wb") as tar_f:
		for index in indices:
			sentence = sentences_raw[index]
			if len(sentence) > max_sent_len:
				continue
			valid_inputs_count += 1
			_lex, _pos = zip(*sentence)
			_lex = list(_lex)
			_pos = list(_pos)
			_lex = lex_tokeniser.texts_to_sequences([_lex])[0]
			_pos = pos_tokeniser.texts_to_sequences([_pos])[0]
			mask = [random() for _ in _lex]
			inp_lx = []
			inp_ps = []
			chance = random() < 0.4
			label = ' '.join(str(tkn) for tkn in make_one_hot(int(chance), 3))
			if chance:
				inp_lx, inp_ps = zip(*sample(list(zip(_lex, _pos)), len(_lex)))
				inp_lx = list(inp_lx)
				inp_ps = list(inp_ps)
			else:
				for i, pr in enumerate(mask):
					if pr > 0.15:
						inp_lx.append(_lex[i])
						inp_ps.append(_pos[i])
					else:
						inp_lx.append(lex_mask)
						inp_ps.append(pos_mask)
			inp_lx = pad_list([lex_start] + inp_lx + [lex_end], seq_pad_len)  # list of tokens
			inp_lx = (pad_list([lex_start] + _lex, seq_pad_len), inp_lx)
			inp_ps = pad_list([pos_start] + inp_ps + [pos_end], seq_pad_len)
			inp_ps = (pad_list([pos_start] + _pos, seq_pad_len), inp_ps)
			tar_lx = pad_list(_lex + [lex_end], seq_pad_len)
			tar_ps = pad_list(_pos + [pos_end], seq_pad_len)
			write_ = delimitter.join([' '.join([str(tkn) for tkn in turn]) for turn in inp_lx])
			# print(write_)
			lex_f.write((write_ + '\n').encode(encoding=encoding))
			write_ = delimitter.join([' '.join([str(tkn) for tkn in turn]) for turn in inp_ps])
			# print(write_)
			pos_f.write((write_ + '\n').encode(encoding=encoding))
			write_ = delimitter.join(["0 1", ' '.join([str(tkn) for tkn in tar_lx]), ' '.join([str(tkn) for tkn in tar_ps]), label])
			# print(write_)
			tar_f.write((write_ + '\n').encode(encoding=encoding))
	print(valid_inputs_count)
	return


def make_tokenisers(sentences):
	lex_tok = tf.keras.preprocessing.text.Tokenizer(num_words=word_vocab, filters='\t\n', split=' ', oov_token='___', lower=True)
	pos_tok = tf.keras.preprocessing.text.Tokenizer(num_words=ptag_vocab, filters='\t\n', split=' ', oov_token='___', lower=True)
	# texts = [list(turn) for turn in switchboard.tagged_turns()]
	# texts = [sent for para in paragraphs for sent in para]
	lex_, pos_ = zip(*(tuple(zip(*sent)) for sent in sentences if sent))
	lex_ = [' '.join(sent) for sent in lex_]
	pos_ = [' '.join(sent) for sent in pos_]
	# del texts
	lex_tok.fit_on_texts(lex_)
	print("finished fitting lexical tokeniser")
	with open(lex_tok_name, "wb") as lex_file:
		pickle.dump(lex_tok, lex_file)
	pos_tok.fit_on_texts(pos_)
	print("finished fitting part of speech tokeniser")
	with open(pos_tok_name, "wb") as pos_file:
		pickle.dump(pos_tok, pos_file)
	print("finished saving tokenisers")
	return


def get_convai(source_name, tagger):
	import re
	paragraphs = []
	current = []
	punctuation = re.compile(r"([A-Za-z])([.,!?])")
	with open(source_name, 'r') as read_f:
		for raw_line in read_f:
			segments = punctuation.sub(r"\1 \2", raw_line).strip().split('\t')
			segments = [item.split() for item in segments]
			index = int(segments[0][0])
			segments = [segments[0][1:]] + segments[1:]
			if index == 1:
				paragraphs.append(current)
				current = []
			current.extend(segments)
	print("finished reading convAI file")
	
	paragraphs = paragraphs[1:]
	for counter in range(len(paragraphs)):
		para_ = paragraphs[counter]
		processed = []
		your_persona = []
		partner_persona = []
		for line in para_:
			if line[0] == "your":
				tags = tagger.tag(line[2:])  # ToDo: this
				pairs = list(zip(line[2:], tags))
				your_persona.extend(pairs)
			elif line[0] == "partner's":
				tags = tagger.tag(line[2:])  # ToDo: this
				pairs = list(zip(line[2:], tags))
				partner_persona.extend(pairs)
			else:
				tags = tagger.tag(line)  # ToDo: this
				pairs = list(zip(line, tags))
				processed.append(pairs)
		if your_persona:
			processed = [your_persona] + processed
		if partner_persona:
			processed = [partner_persona] + processed
		paragraphs[counter] = processed
	print("finished processing convAI file")
	return paragraphs


def get_switchboard():
	from nltk.corpus import switchboard
	return [[list(sent) for sent in disc] for disc in switchboard.tagged_discourses()]


def get_brown():
	from nltk.corpus import brown
	return brown.tagged_paras()


def get_brown_universal():
	from nltk.corpus import brown
	return brown.tagged_paras(tagset='universal')


def main():
	# rand_tag = ConsistentRandomTagger.load("dummy_tagger.pickle")
	# convai_dir = "C:/Users/admin/Downloads/convai2text/_/"
	# convai_files = ["valid_both_original_no_cands.txt", "valid_none_original_no_cands.txt", "valid_both_revised_no_cands.txt",
	# 				"train_both_original_no_cands.txt", "train_none_original_no_cands.txt", "train_both_revised_no_cands.txt"]
	# convai_file = convai_dir + convai_files[2]
	# paragraphs_raw = get_convai(convai_file, rand_tag)
	
	# print(paragraphs_raw[0][0])
	# print(paragraphs_raw[0][1])
	# print(paragraphs_raw[0][2])
	# print(paragraphs_raw[0][3])
	paragraphs_raw = nltk.corpus.brown.tagged_sents()
	# make_tokenisers(paragraphs_raw)
	# make_cola()
	make_mlm_inputs(paragraphs_raw)
	# make_input_files(paragraphs_raw)
	

if __name__ == '__main__':
	lex_tok_name = "pickle_jar/brown_tok_lex_15000.pickle"
	pos_tok_name = "pickle_jar/brown_tok_pos_500.pickle"
	data_name = "text_data/brown_15000_500"
	delimitter = " ||| "
	encoding = 'ascii'
	
	word_vocab = 15000
	ptag_vocab = 500
	
	max_sent_len = 95  # will be extended by 2 places (start and end tokens), effective length = 75
	overlap = 0
	seq_pad_len = 100
	main()
