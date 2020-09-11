import nltk
import re
#
bnc = nltk.corpus.reader.BNCCorpusReader(root="C:/Users/admin/PycharmProjects/Transformer/BNC/Texts", fileids=r'[A-K]/\w*/\w*\.xml')
# files = bnc.fileids()
# test = bnc.tagged_sents(fileids=files[-1], c5=True)
# print(len(files))
#
jabber = """Twas brillig, and the slithy toves Did gyre and gimble in the wabe; All mimsy were the borogoves, And the mome raths outgrabe.
“Beware the Jabberwock, my son! The jaws that bite, the claws that catch! Beware the Jubjub bird, and shun The frumious Bandersnatch!”
He took his vorpal sword in hand: Long time the manxome foe he sought— So rested he by the Tumtum tree, And stood awhile in thought.
And as in uffish thought he stood, The Jabberwock, with eyes of flame, Came whiffling through the tulgey wood, And burbled as it came!
One, two! One, two! And through and through The vorpal blade went snicker-snack! He left it dead, and with its head He went galumphing back.
“And hast thou slain the Jabberwock? Come to my arms, my beamish boy! O frabjous day! Callooh! Callay!” He chortled in his joy.
"""
punctuation = re.compile(r"(.)([.,:;\"!?])")
punctuation2 = re.compile(r"([.,:;\"“!?])(.)")
jabber = jabber.split('\n')
jabber = [punctuation2.sub(r"\1 \2", punctuation.sub(r"\1 \2", stanza)).lower().split() for stanza in jabber]
brown = nltk.corpus.brown
brn_files = brown.fileids()
train = brown.tagged_sents()[:-2000]
test = brown.tagged_sents()[-2000:]


def tagger_stats(tgr, jbr=False):
	# print(tgr.evaluate(train))
	print(tgr.evaluate(test))
	if jbr:
		for stanza in jabber:
			if not stanza:
				continue
			w_, t_ = zip(*[(str(w), str(t)) for w, t in tgr.tag(stanza)])
			print(' '.join(w_))
			print(' '.join(t_))


uni_tagger = nltk.tag.sequential.UnigramTagger(model=dict(nltk.corpus.brown.tagged_words()))
print("unigram")
tagger_stats(uni_tagger)
bi_tagger = nltk.tag.sequential.BigramTagger(train=train, backoff=uni_tagger)
print("bigram")
tagger_stats(bi_tagger)
tri_tagger = nltk.tag.sequential.TrigramTagger(train=train, backoff=uni_tagger)
print("trigram - unigram")
tagger_stats(tri_tagger)
tri_tagger = nltk.tag.sequential.TrigramTagger(train=train, backoff=bi_tagger)
print("trigram - bigram")
tagger_stats(tri_tagger)
# crf_tagger = nltk.tag.crf.CRFTagger()
# crf_tagger.train(train, 'tagger_brown_crf.crf.tagger')
# print("crf")
# # tagger_stats(crf_tagger)
# gram4_tagger = nltk.tag.sequential.NgramTagger(4, train=train, backoff=uni_tagger)
# print("4 gram")
# tagger_stats(gram4_tagger)
# gram4_tagger = nltk.tag.sequential.NgramTagger(4, train=train, backoff=bi_tagger)
# print("4 gram - bigram")
# tagger_stats(gram4_tagger)
# gram4_tagger = nltk.tag.sequential.NgramTagger(4, train=train, backoff=tri_tagger)
# print("4 gram - trigram")
# tagger_stats(gram4_tagger)
from taggers import feature_detector, BayesTagger, StackTagger
bayes = BayesTagger(feature_detector=feature_detector, train=train)
print("bayes tagger no back off")
tagger_stats(bayes, jbr=True)
stack = StackTagger(main_tagger=bi_tagger, other_tagger=bayes)
print("stack tagger")
tagger_stats(stack, jbr=True)
bayes2 = BayesTagger(feature_detector=feature_detector, train=train, backoff=bi_tagger)
print("bayes tagger, bigram back off")
tagger_stats(bayes2, jbr=True)
# from taggers import AgnosticSurroundTagger
# agn_tagger = AgnosticSurroundTagger(1, train=train, backoff=tri_tagger)
# print("agnostic")
# tagger_stats(agn_tagger, True)
# agn_2_tagger = AgnosticSurroundTagger(2, train=train, backoff=agn_tagger)
# print("agnostic 2")
# tagger_stats(agn_2_tagger, True)
# agn_3_tagger = AgnosticSurroundTagger(3, train=train, backoff=agn_2_tagger)
# print("agnostic 3")
# tagger_stats(agn_3_tagger, True)
