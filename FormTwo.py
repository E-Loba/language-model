# expected format of config file:
##################################
# _input LEXICAL FILE PATH_ 0
# _input POS TAG FILE PATH_ 1
# _input INDICES FILE PATH_ 2
# _SEQ LEN_  ___  _SEN LEN_ 3
# _BATCH SIZE_				4
# _NUMBER OF EPOCHS_		5
# _word transformer HYPER-PARAMETERS_: LAYERS-number EMBEDDING-dimension attention-HEADS hidden-DIMENSION learn-RATE
# _checkpoint FOLDER name_	7
# _architecture name_		8
##################################
# each setting is line separated
# hyper-parameters are space separated on one line
from layer_definitions import *


class FormTwo(tf.keras.Model):
	def __init__(self, layers, dim, pos_dim, dff, heads, vocab, pos_vocab, rate=0.1, max_len=1000):
		super(FormTwo, self).__init__()
		self.word_embeddings = tf.keras.layers.Embedding(vocab, dim)
		self.pos_embeddings = tf.keras.layers.Embedding(pos_vocab, pos_dim)

		self.word_position_encoding = positional_encoding(max_len, dim)
		self.pos_position_encoding = positional_encoding(max_len, pos_dim)

		self.word_encoder = Encoder(layers, dim, heads, dff, self.word_embeddings, self.word_position_encoding, rate)
		self.pos_encoder = Encoder(layers, pos_dim, heads, dff, self.pos_embeddings, self.pos_position_encoding, rate)
		# self.pos_to_words_prepare = tf.keras.layers.Dense(dim)
		self.pos_to_words = RawStack(layers, dim, heads, dff, rate)

		self.pos_logits = tf.keras.layers.Dense(pos_vocab)  # pos prediction logits
		self.final = tf.keras.layers.Dense(vocab)  # word prediction logits
		self.predict = tf.keras.Sequential([tf.keras.layers.LayerNormalization(epsilon=1e-6), tf.keras.layers.Flatten(), tf.keras.layers.Dense(3, activation='softmax')])
		# predict whether is next sentence or not
	
	def get_trainable_vars(self):
		pos_vars = self.pos_embeddings.trainable_variables + self.pos_encoder.trainable_variables + self.pos_logits.trainable_variables
		lex_vars = self.word_embeddings.trainable_variables + self.word_encoder.trainable_variables + self.pos_to_words.trainable_variables + self.final.trainable_variables
		# + self.pos_to_words_prepare.trainable_variables
		predict_vars = self.predict.trainable_variables
		return pos_vars, lex_vars, predict_vars

	def _encode(self, contexts_, turn_labels, training, encoder):
		contexts = []
		masks = []
		offset = 0
		num = len(contexts_)
		for turn, context in zip(turn_labels, contexts_):
			mask_ = create_padding_mask(context)
			context, seq_len = encoder([context], training, [mask_], offset, turn)
			masks.append(mask_)
			contexts.append(context)
			offset += seq_len + 1
		return contexts, masks, offset

	def call(self, inputs, turn_labels=None, training=None, mask=None):
		"""
		:param inputs: tuple( words, pos )
						words = list [target, history...]
						pos = list [target, history...]
				history -> most ancient ... most recent
		:param turn_labels:
		:param training:
		:param mask: compute here
		:return:
		"""
		word_contexts, pos_contexts = inputs
		word_tar = word_contexts[0]
		word_contexts = word_contexts[1:]
		pos_tar = pos_contexts[0]
		pos_contexts = pos_contexts[1:]

		pos_contexts, pos_masks, offset = self._encode(pos_contexts, turn_labels, training, self.pos_encoder)
		word_contexts, masks, offset = self._encode(word_contexts, turn_labels, training, self.word_encoder)
		mix_contexts = []
		for x, y, m in zip(pos_contexts, word_contexts, pos_masks):
			context = self.pos_to_words((x, y), training, m)
			mix_contexts.append(context)

		mask_ = tf.maximum(create_padding_mask(pos_tar), create_look_ahead_mask(tf.shape(pos_tar)[1]))
		pos_prediction, _ = self.pos_encoder([pos_tar] + pos_contexts, training, [mask_] + pos_masks, offset, 0)
		pos_outp = self.pos_logits(pos_prediction)
		target_encoding, _ = self.word_encoder([word_tar], training, [mask_], offset, 0)
		# pos_prediction = self.pos_to_words_prepare(pos_prediction)
		
		mask_ = create_padding_mask(pos_tar)
		for c, m in zip(mix_contexts + [pos_prediction], pos_masks + [mask_]):
			pos_prediction = self.pos_to_words((c, target_encoding), training, m)
		word_outp = self.final(pos_prediction)
		prediction = self.predict(tf.concat([word_outp, pos_outp], axis=-1))
		return word_outp, pos_outp, prediction


class DualLongFormTwo(FormTwo):
	def __init__(self, layers, dim, pos_dim, dff, heads, vocab, pos_vocab, rate=0.1, max_len=1000):
		super(DualLongFormTwo, self).__init__(layers, dim, pos_dim, dff, heads, vocab, pos_vocab, rate=rate, max_len=max_len)
		# del self.pos_to_words_prepare

	def call(self, inputs, turn_labels=None, training=None, mask=None):
		"""
		:param inputs: tuple( words, pos )
						words = list [target, history...]
						pos = list [target, history...]
				history -> most ancient ... most recent
		:param turn_labels:
		:param training:
		:param mask: compute here
		:return:
		"""
		word_contexts = inputs[0]
		pos_contexts = inputs[1]
		pos_tar = pos_contexts[0]
		pos_contexts = pos_contexts[1:]
		word_tar = word_contexts[0]
		word_contexts = word_contexts[1:]

		pos_contexts, pos_masks, pos_offset = self._encode(pos_contexts, turn_labels, training, self.pos_encoder)
		word_contexts, masks, word_offset = self._encode(word_contexts, turn_labels, training, self.word_encoder)
		mix_contexts = []
		for x, y, m in zip(pos_contexts, word_contexts, pos_masks):
			context = self.pos_to_words((x, y), training, m)
			mix_contexts.append(context)

		mask_ = tf.maximum(create_padding_mask(word_tar), create_look_ahead_mask(tf.shape(word_tar)[1]))
		word_prediciton, _ = self.word_encoder([word_tar] + word_contexts, training, [mask_] + masks, word_offset, 0)

		mask_ = tf.maximum(create_padding_mask(pos_tar), create_look_ahead_mask(tf.shape(pos_tar)[1]))
		pos_prediction, _ = self.pos_encoder([pos_tar] + pos_contexts, training, [mask_] + pos_masks, pos_offset, 0)
		pos_outp = self.pos_logits(pos_prediction)

		final_prediction = self.pos_to_words((pos_prediction, word_prediciton), training, mask_)
		for c, m in zip(mix_contexts, pos_masks):
			final_prediction = self.pos_to_words((c, final_prediction), training, m)
		word_outp = self.final(final_prediction)
		prediction = self.predict(tf.concat([word_outp, pos_outp], axis=-1))
		return word_outp, pos_outp, prediction


class DualShortFormTwo(FormTwo):
	def __init__(self, layers, dim, pos_dim, dff, heads, vocab, pos_vocab, rate=0.1, max_len=1000):
		super(DualShortFormTwo, self).__init__(layers, dim, pos_dim, dff, heads, vocab, pos_vocab, rate=rate, max_len=max_len)
		# del self.pos_to_words_prepare

	def call(self, inputs, turn_labels=None, training=None, mask=None):
		"""
		:param inputs: tuple( words, pos )
						words = list [target, history...]
						pos = list [target, history...]
				history -> most ancient ... most recent
		:param turn_labels:
		:param training:
		:param mask: compute here
		:return:
		"""
		word_contexts = inputs[0]
		pos_contexts = inputs[1]
		pos_tar = pos_contexts[0]
		pos_contexts = pos_contexts[1:]
		word_tar = word_contexts[0]
		word_contexts = word_contexts[1:]

		pos_contexts, pos_masks, pos_offset = self._encode(pos_contexts, turn_labels, training, self.pos_encoder)
		word_contexts, masks, word_offset = self._encode(word_contexts, turn_labels, training, self.word_encoder)

		mask_ = tf.maximum(create_padding_mask(word_tar), create_look_ahead_mask(tf.shape(word_tar)[1]))
		word_prediciton, _ = self.word_encoder([word_tar] + word_contexts, training, [mask_] + masks, word_offset, 0)

		mask_ = tf.maximum(create_padding_mask(pos_tar), create_look_ahead_mask(tf.shape(pos_tar)[1]))
		pos_prediction, _ = self.pos_encoder([pos_tar] + pos_contexts, training, [mask_] + pos_masks, pos_offset, 0)
		pos_outp = self.pos_logits(pos_prediction)

		final_prediction = self.pos_to_words((pos_prediction, word_prediciton), training, mask_)
		word_outp = self.final(final_prediction)
		mix_contexts = []
		for x, y, m in zip(pos_contexts, word_contexts, pos_masks):
			context = self.pos_to_words((x, y), training, m)
			mix_contexts.append(context)
		prediction = self.predict(tf.concat([word_outp, pos_outp], axis=-1))
		return word_outp, pos_outp, prediction


class MixedFormTwo(FormTwo):
	def __init__(self, layers, dim, pos_dim, dff, heads, vocab, pos_vocab, rate=0.1, max_len=1000):
		super(MixedFormTwo, self).__init__(layers, dim, pos_dim, dff, heads, vocab, pos_vocab, rate=rate, max_len=max_len)
		# del self.pos_to_words_prepare

	def call(self, inputs, turn_labels=None, training=None, mask=None):
		"""
		:param inputs: tuple( words, pos )
						words = list [target, history...]
						pos = list [target, history...]
				history -> most ancient ... most recent
		:param turn_labels:
		:param training:
		:param mask: compute here
		:return:
		"""
		word_contexts = inputs[0]
		word_contexts = word_contexts[1:] + [word_contexts[0]]  # put most recent input at the end
		pos_contexts = inputs[1]
		pos_contexts = pos_contexts[1:] + [pos_contexts[0]]  # put most recent input at the end

		pos_contexts, pos_masks, pos_offset = self._encode(pos_contexts, turn_labels, training, self.pos_encoder)
		word_contexts, masks, word_offset = self._encode(word_contexts, turn_labels, training, self.word_encoder)

		contexts = []
		output = self.pos_to_words((pos_contexts[0], word_contexts[0]), training, pos_masks[0])
		contexts += [output]
		for x, y, m in zip(pos_contexts[1:], word_contexts[1:], pos_masks[1:]):
			context = self.pos_to_words((x, y), training, m)
			output = self.pos_to_words((context, output), training, m)
			contexts += [context]

		word_outp = self.final(output)
		pos_outp = self.pos_logits(output)
		prediction = self.predict(tf.concat([word_outp, pos_outp], axis=-1))
		return word_outp, pos_outp, prediction
