import tensorflow as tf
import re


def prepare_raw_string(string):
	pre_punctuation = re.compile(r"([^\w'\s])([A-Za-z])")
	post_punctuation = re.compile(r"([A-Za-z])([^\w'\s])")
	outp = pre_punctuation.sub(r"\1 \2", string)
	outp = post_punctuation.sub(r"\1 \2", outp)
	return outp.lower().split()


def make_one_hot(index, length):
	return [1 if index == _ else 0 for _ in range(length)]


def str2int(string):
	return [int(token) for token in string.split()]


def str2tensor(string):
	return tf.expand_dims(tf.convert_to_tensor(str2int(string), dtype=tf.int32), axis=0)


def pad_list(l_, pad_len):
	padding = pad_len - len(l_)
	if padding < 0:
		raise ValueError
	else:
		return l_ + [0] * padding


def truncate_split(my_list, n, o=0):
	for i in range(o, len(my_list), n-o):
		yield my_list[i-o:i + n-o]


def pad2len(t_, len_):
	size = len_ - tf.shape(t_)[1]
	paddings = [[0, 0], [0, size]]
	return tf.pad(t_, paddings)
