import random
import pickle
import nltk


class StackTagger(nltk.tag.api.TaggerI):
	def __init__(self, main_tagger=None, other_tagger=None):
		self.main_tagger = main_tagger
		self.other = other_tagger
	
	def tag(self, tokens):
		tkns, tags = zip(*self.main_tagger.tag(tokens))
		return self.other.tag(tkns, tags)


class BayesTagger(nltk.tag.sequential.ClassifierBasedTagger):
	def tag(self, tokens, tags=None):
		outp = []
		for i in range(len(tokens)):
			if not tags:
				outp.append(self.tag_one(tokens, i, outp))
			elif not tags[i]:
				outp.append(self.tag_one(tokens, i, tags))
			else:
				outp.append(tags[i])
		return list(zip(tokens, outp))
	
	def _train(self, tagged_corpus, classifier_builder, verbose):
		"""
		Build a new classifier, based on the given training data
		*tagged_corpus*.
		"""
		classifier_corpus = []
		if verbose:
			print('Constructing training corpus for classifier.')
		for sentence in tagged_corpus:
			untagged_sentence, tags = zip(*sentence)
			for index in range(len(sentence)):
				featureset = self.feature_detector(untagged_sentence, index, tags)
				classifier_corpus.append((featureset, tags[index]))
		if verbose:
			print('Training classifier (%d instances)' % len(classifier_corpus))
		self._classifier = classifier_builder(classifier_corpus)


def feature_detector(tokens, index, history):
	outp = {}
	outp['tkn'] = tokens[index]
	outp['prev tkn'] = -1 if index <= 0 else tokens[index - 1]
	outp['prev tag'] = -1 if index <= 0 else history[-1]
	outp['pre prev tkn'] = -1 if index <= 1 else tokens[index - 2]
	outp['pre prev tag'] = -1 if index <= 1 else history[-2]
	outp['next tkn'] = -2 if index + 1 >= len(tokens) else tokens[index + 1]
	outp['ne next tkn'] = -2 if index + 2 >= len(tokens) else tokens[index + 2]
	outp['next tag'] = -2 if index + 1 >= len(history) else history[index+1]
	outp['ne next tag'] = -2 if index + 2 >= len(history) else history[index+2]
	outp['prefix'] = tokens[index][:4]
	outp['suffix'] = tokens[index][-4:]
	outp['len'] = len(tokens[index])
	return outp
# ToDo: implement taggers


def prepare_bnc_tagger():
	from nltk.corpus.reader import BNCCorpusReader
	bnc = BNCCorpusReader(root="C:/Users/admin/PycharmProjects/Transformer/BNC/Texts", fileids=r'[A-K]/\w*/\w*\.xml')
	tagger_ = make_my_tagger(bnc.tagged_sents(c5=True))
	print("trained c5 tagger")
	with open("tagger_bnc.pickle", "wb") as f_:
		pickle.dump(tagger_, f_)
	del tagger_
	tagger_ = make_my_tagger(bnc.tagged_sents())
	print("trained bnc universal tagger")
	with open("tagger_universal_bnc.pickle", "wb") as f_:
		pickle.dump(tagger_, f_)
	return


class SurroundTagger(nltk.tag.sequential.NgramTagger):
	def tag(self, tokens):
		stack = self._taggers[1:]
		tags = [t for w, t in stack[0].tag(tokens)] if stack else []
		for i in range(len(tokens)):
			if not tags[i]:
				tags[i] = self.tag_one(tokens, i, tags)
		return list(zip(tokens, tags))
	
	def tag_one(self, tokens, index, context_):
		return self.choose_tag(tokens, index, context_)
	
	def context(self, tokens, index, context_):
		"""
		
		:param tokens:
		:param index:
		:param context_: unlike history, it's the full length of the token sequence
		:return:
		"""
		tag_context = tuple(context_[max(0, index - self._n): index])
		tag_context += tuple(context_[index + 1:index + self._n + 1])
		return tag_context, tokens[index]


class AgnosticSurroundTagger(nltk.tag.sequential.NgramTagger):
	def tag(self, tokens):
		stack = self._taggers[1:]
		tags = [t for w, t in stack[0].tag(tokens)] if stack else []
		for i in range(len(tokens)):
			if not tags[i]:
				tags[i] = self.tag_one(tokens, i, tags)
		return list(zip(tokens, tags))
	
	def tag_one(self, tokens, index, context_):
		return self.choose_tag(tokens, index, context_)
	
	def context(self, tokens, index, context_):
		"""

		:param tokens:
		:param index:
		:param context_: unlike history, it's the full length of the token sequence
		:return:
		"""
		tag_context = tuple(context_[max(0, index - self._n): index])
		# tag_context += tuple(context_[index + 1:index + 2])
		return tag_context


class AgnosticBackTagger(nltk.tag.sequential.NgramTagger):
	def context(self, tokens, index, history):
		tag_context = tuple(history[max(0, index - self._n): index])
		return tag_context


class RandomTagger:
	def save(self, my_name):
		with open(my_name, "wb") as dump_file:
			pickle.dump(self, dump_file)
	
	@classmethod
	def load(cls, my_name):
		with open(my_name, "rb") as load_file:
			contents = pickle.load(load_file)
		return contents


class TrueRandomTagger(RandomTagger):
	def __init__(self, tag_vocab_size, weights=None):
		if type(tag_vocab_size) != int:
			raise TypeError
		self._tags = random.choices(list(range(2, tag_vocab_size + 2)), weights, k=10000)
	
	def tag(self, sequence):
		return ["tag_" + str(random.choice(self._tags)) for _ in sequence]


class ConsistentRandomTagger(RandomTagger):
	def __init__(self, text, tag_vocab_size):
		if type(tag_vocab_size) != int:
			raise TypeError
		_tags = list(range(2, tag_vocab_size + 2))
		vocab_set = set(text)
		self._mapping = {word.lower(): random.choice(_tags) for word in vocab_set}
	
	def tag(self, sequence):
		assert self._mapping
		output = []
		for word in sequence:
			try:
				tag_ = self._mapping[word.lower()]
			except KeyError:
				tag_ = -1
			output.append("tag_" + str(tag_))
		return output


class RandomErrorTagger(RandomTagger):
	def __init__(self, text, tag_vocab_size, error_prob):
		if type(tag_vocab_size) != int:
			raise TypeError
		self._tags = list(range(2, tag_vocab_size + 2))
		self._p = error_prob
		vocab_set = set(text)
		self._mapping = {word.lower(): random.choice(self._tags) for word in vocab_set}
	
	def tag(self, sequence):
		assert self._mapping
		output = []
		for word in sequence:
			try:
				tag_ = self._mapping[word.lower()]
			except KeyError:
				tag_ = -1
			if random.random() < self._p:
				tag_ = random.choice(self._tags)
			output.append("tag_" + str(tag_))
		return output


def make_my_tagger(data):
	unique_tagger = nltk.tag.sequential.UnigramTagger(train=data)
	bigram_tagger = nltk.tag.sequential.BigramTagger(train=data, backoff=unique_tagger)
	# agnostic_1_tagger = AgnosticBackTagger(1, train=data, backoff=bigram_tagger)
	# agnostic_2_tagger = AgnosticBackTagger(2, train=data, backoff=agnostic_1_tagger)
	# surround_1_tagger = AgnosticSurroundTagger(3, train=data, backoff=bigram_tagger)
	# surround_2_tagger = AgnosticSurroundTagger(2, train=data, backoff=surround_1_tagger)
	final_tagger = AgnosticSurroundTagger(1, train=data, backoff=bigram_tagger)
	return final_tagger


def test_my_tagger():
	import re
	data = list(nltk.corpus.brown.tagged_sents())
	train = data[0::3] + data[1::3]
	eval = data[2::3]
	tagger_ = make_my_tagger(train)
	print("stack tagger")
	print(tagger_.evaluate(train))
	print(tagger_.evaluate(eval))
	print(tagger_.evaluate(data))
	jabber = "’Twas brillig, and the slithy toves Did gyre and gimble in the wabe; All mimsy were the borogoves, And the mome raths outgrabe."
	punctuation = re.compile(r"([A-Za-z])([.,:;!?])")
	jabber = punctuation.sub(r"\1 \2", jabber).lower().split()
	print(' '.join([str(t) for w, t in tagger_.tag(jabber)]))
	return tagger_
	

def train_tagger():
	convai_file = "C:/Users/admin/Downloads/convai2text/_/valid_both_original_no_cands.txt"
	text_data = []
	with open(convai_file, 'r') as f_:
		for line in f_:
			text_data.extend(line.split())
	dummy_tagger = ConsistentRandomTagger(text_data, tag_number)
	dummy_tagger.save(save_name)


if __name__ == '__main__':
	tag_number = 20
	save_name = "dummy_tagger.pickle"
	# prepare_bnc_tagger()
	text = [[(w.lower(), t.split('-')[0]) for w, t in sent] for sent in nltk.corpus.brown.tagged_sents()]
	uni_tagger = nltk.tag.sequential.UnigramTagger(train=text)
	bi_tagger = nltk.tag.sequential.BigramTagger(train=text, backoff=uni_tagger)
	print("made bigram tagger")
	bayes = BayesTagger(feature_detector=feature_detector, train=text)
	print("made bayes tagger")
	stack = StackTagger(main_tagger=bi_tagger, other_tagger=bayes)
	with open("pickle_jar/brown_full_tagger.pickle", "wb") as write_:
		pickle.dump(stack, write_)
	print("saved stack tagger")
