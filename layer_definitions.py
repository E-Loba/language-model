import tensorflow as tf
import numpy as np


def get_angles(pos, i, d_model):
	angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
	return pos * angle_rates


def segment_embedding(item, segment_id):
	# ToDo: test effect
	segment_encoding = tf.zeros_like(item)
	segment_encoding += segment_id
	return segment_encoding


def positional_encoding(position, d_model):
	angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)

	# apply sin to even indices in the array; 2i
	sines = np.sin(angle_rads[:, 0::2])
	# apply cos to odd indices in the array; 2i+1
	cosines = np.cos(angle_rads[:, 1::2])
	pos_encoding = np.concatenate([sines, cosines], axis=-1)
	pos_encoding = pos_encoding[np.newaxis, ...]
	return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
	seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

	# add extra dimensions so that we can add the padding
	# to the attention logits.
	return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
	mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
	return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
	"""Calculate the attention weights.
	q, k, v must have matching leading dimensions.
	k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
	The mask has different shapes depending on its type(padding or look ahead)
	but it must be broadcastable for addition.

	Args:
	q: query shape == (..., seq_len_q, depth)
	k: key shape == (..., seq_len_k, depth)
	v: value shape == (..., seq_len_v, depth_v)
	mask: Float tensor with shape broadcastable
			to (..., seq_len_q, seq_len_k). Defaults to None.

	Returns:
	output, attention_weights
	"""
	# print("attention q shape: {}".format(tf.shape(q)))
	# print("attention k shape: {}".format(tf.shape(k)))
	# print("attention v shape: {}".format(tf.shape(v)))
	matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
	# print("attention q*k shape: {}".format(tf.shape(matmul_qk)))
	# scale matmul_qk
	dk = tf.cast(tf.shape(k)[-1], tf.float32)
	scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
	# print("attention logits shape: {}".format(tf.shape(scaled_attention_logits)))
	# print("attention mask shape: {}".format(tf.shape(mask)))
	# add the mask to the scaled tensor.
	if mask is not None:
		scaled_attention_logits += (mask * -1e9)
	# print("attention masked logits shape: {}".format(tf.shape(scaled_attention_logits)))
	# softmax is normalized on the last axis (seq_len_k) so that the scores
	# add up to 1.
	attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
	# print("attention weights shape: {}".format(tf.shape(attention_weights)))
	output = tf.matmul(attention_weights, v)  # (..., seq_len_v, depth_v)
	# print("attention output shape: {}".format(tf.shape(output)))
	return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
	def __init__(self, d_model, num_heads):
		super(MultiHeadAttention, self).__init__()
		self.num_heads = num_heads
		self.d_model = d_model
		assert d_model % self.num_heads == 0
		self.depth = d_model // self.num_heads
		self.wq = tf.keras.layers.Dense(d_model)
		self.wk = tf.keras.layers.Dense(d_model)
		self.wv = tf.keras.layers.Dense(d_model)
		self.dense = tf.keras.layers.Dense(d_model)

	def split_heads(self, x, batch_size):
		"""Split the last dimension into (num_heads, depth).
		Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
		"""
		x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
		return tf.transpose(x, perm=[0, 2, 1, 3])

	def call(self, v, k, q, mask):
		batch_size = tf.shape(q)[0]

		q = self.wq(q)  # (batch_size, seq_len, d_model)
		k = self.wk(k)  # (batch_size, seq_len, d_model)
		v = self.wv(v)  # (batch_size, seq_len, d_model)

		q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
		k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
		v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
		# print("multi-head attention q shape: {}".format(tf.shape(q)))
		# print("multi-head attention k shape: {}".format(tf.shape(k)))
		# print("multi-head attention v shape: {}".format(tf.shape(v)))
		# scaled_attention.shape == (batch_size, num_heads, seq_len_v, depth)
		# attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
		scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
		# print("multi-head attention scaled_attention shape: {}".format(tf.shape(scaled_attention)))
		scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
		# print("multi-head attention transpose shape: {}".format(tf.shape(scaled_attention)))
		# (batch_size, seq_len_v, num_heads, depth)
		concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
		# print("multi-head attention concatenated shape: {}".format(tf.shape(concat_attention)))
		# (batch_size, seq_len_v, d_model)
		output = self.dense(concat_attention)  # (batch_size, seq_len_v, d_model)
		# print("multi-head attention output shape: {}".format(tf.shape(output)))
		return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
	# (batch_size, seq_len, dff)
	# (batch_size, seq_len, d_model)
	return tf.keras.Sequential([tf.keras.layers.Dense(dff, activation='relu'), tf.keras.layers.Dense(d_model)])


class RawEncoder(tf.keras.layers.Layer):
	def __init__(self, d_model, num_heads, dff, rate=0.1):
		super(RawEncoder, self).__init__()
		self.mha = MultiHeadAttention(d_model, num_heads)
		self.ffn = point_wise_feed_forward_network(d_model, dff)
		self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.dropout1 = tf.keras.layers.Dropout(rate)
		self.dropout2 = tf.keras.layers.Dropout(rate)

	def call(self, x, training=None, mask=None):
		"""
		:param x: tuple ( pos_context, word_context )
		:param training:
		:param mask: pos context mask
		:return:
		"""
		# print("encode layer x shape: {}".format(tf.shape(x)))
		pos_, x_ = x
		attn_output, _ = self.mha(pos_, pos_, x_, mask)  # (batch_size, input_seq_len, d_model)

		# print("encode layer attention shape: {}".format(tf.shape(attn_output)))
		attn_output = self.dropout1(attn_output, training=training)
		out1 = self.layernorm1(x_ + attn_output)  # (batch_size, input_seq_len, d_model)
		# print("encode layer x + attention shape: {}".format(tf.shape(out1)))
		ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
		ffn_output = self.dropout2(ffn_output, training=training)
		out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
		# print("encode layer output shape: {}".format(tf.shape(out2)))
		return out2


class RawStack(tf.keras.layers.Layer):
	def __init__(self, layers, dim, heads, dff, rate=0.1):
		super(RawStack, self).__init__()
		self.raw_layers = [RawEncoder(dim, heads, dff, rate) for _ in range(layers)]
		self.layers_num = layers

	def call(self, inputs, training=None, mask=None):
		"""
		:param inputs: tuple ( pos_context, word_context )
		:param training:
		:param mask: pos context mask
		:return:
		"""
		x, y = inputs
		for _ in range(self.layers_num):
			outp = self.raw_layers[_]((x, y), training, mask)
			y = outp
		return y


class EncoderLayer(tf.keras.layers.Layer):
	def __init__(self, d_model, num_heads, dff, rate=0.1):
		super(EncoderLayer, self).__init__()
		self.mha = MultiHeadAttention(d_model, num_heads)
		self.ffn = point_wise_feed_forward_network(d_model, dff)
		self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.dropout1 = tf.keras.layers.Dropout(rate)
		self.dropout2 = tf.keras.layers.Dropout(rate)

	def call(self, x, training=None, mask=None):
		# print("encode layer x shape: {}".format(tf.shape(x)))
		tar = x[0]
		attn = 0.0
		num = len(x)
		for inp, mask_ in zip(x, mask):
			step, _ = self.mha(inp, inp, tar, mask_)
			step = tf.divide(step, num)
			attn = tf.add(attn, step)

		# print("encode layer attention shape: {}".format(tf.shape(attn_output)))
		attn = self.dropout1(attn, training=training)
		out1 = self.layernorm1(tar + attn)  # (batch_size, input_seq_len, d_model)
		# print("encode layer x + attention shape: {}".format(tf.shape(out1)))
		ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
		ffn_output = self.dropout2(ffn_output, training=training)
		out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
		# print("encode layer output shape: {}".format(tf.shape(out2)))
		return [out2] + x[1:]


class Encoder(tf.keras.layers.Layer):
	def __init__(self, num_layers, d_model, num_heads, dff, embeddings, position_encoding, rate=0.1):
		super(Encoder, self).__init__()
		self.d_model = d_model
		self.embedding = embeddings
		self.num_layers = num_layers
		self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
		self.dropout = tf.keras.layers.Dropout(rate)
		self.position_encoding = position_encoding  # ToDo

	def call(self, x, training=None, mask=None, offset=0, turn=0):
		elem = x[0]
		seq_len = tf.shape(elem)[1]
		x_ = self.embedding(elem)
		x_ *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
		x_ += segment_embedding(x_, turn)
		# print("encoder segment embedding shape: {}".format(tf.shape(x)))
		x_ += self.position_encoding[:, offset:offset + seq_len, :]
		# print("encoder positional encoding shape: {}".format(tf.shape(x)))
		x_ = self.dropout(x_, training=training)
		x = [x_] + x[1:]
		for i in range(self.num_layers):
			x = self.enc_layers[i](x, training, mask)
		# print("encoder output shape: {}".format(tf.shape(x)))
		return x[0], seq_len  # (batch_size, input_seq_len, d_model)