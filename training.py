import FormTwo
import numpy as np
from gzip import open as zopen
from z_util import *


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
	def __init__(self, d_model, warmup_steps=4000):
		super(CustomSchedule, self).__init__()
		self.d_model = d_model
		self.d_model = tf.cast(self.d_model, tf.float32)
		self.warmup_steps = warmup_steps
	
	def __call__(self, step):
		arg1 = tf.math.rsqrt(step)
		arg2 = step * (self.warmup_steps**-1.5)
		return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def f1_score(result, truth):
	precision = len(set(result.numpy()).intersection(set(truth.numpy()))) / len(result.numpy())
	recall = len(set(truth.numpy()).intersection(set(result.numpy()))) / len(truth.numpy())
	return 2 * (precision * recall) / (precision + recall)


def make_hyper_parameters(hp_list):
	return {"layers": int(hp_list[0]), "dim": int(hp_list[1]), "pos_dim": int(hp_list[2]), "dff": int(hp_list[3]),
			"heads": int(hp_list[4]), "vocab": int(hp_list[5]), "pos_vocab": int(hp_list[6]), "rate": float(hp_list[7])}


def make_global_vars(configs):
	input_lex_file = configs[0].strip()
	input_pos_file = configs[1].strip()
	input_ind_file = configs[2].strip()
	max_length = int(configs[3])
	batch_size = int(configs[4])
	epochs = int(configs[5])
	hp_lex = make_hyper_parameters([hp.strip() for hp in configs[6].split(' ')])
	chkpt_dir = configs[7].strip()
	architecture = configs[8].strip()
	delimitter = configs[9].strip()
	encoding = configs[10].strip()
	return input_lex_file, input_pos_file, input_ind_file, max_length, batch_size, epochs, hp_lex, chkpt_dir, architecture, delimitter, encoding


def main():
	from time import time
	from sys import argv
	import csv
	
	config_file = argv[1]
	configs = []
	with open(config_file, "r") as config_file_:
		for line in config_file_:
			configs.append(line)
	available_models = {"FormTwo": FormTwo.FormTwo, "DualLongFormTwo": FormTwo.DualLongFormTwo, "DualShortFormTwo": FormTwo.DualShortFormTwo, "MixedFormTwo": FormTwo.MixedFormTwo}
	lex_file_name, pos_file_name, tar_file_name, max_length, batch_size, epochs, hp_lex, chkpt_dir, architecture, delimitter, encoding = make_global_vars(configs)
	try:
		architecture = available_models[architecture]
	except KeyError as kE:
		print(kE)
		print()
		print("##############################################################\n")
		print("invalid architecture name. Please choose one of the following:")
		print(list(available_models.keys()))
		print()
		print("##############################################################\n")
		exit(1)
	
	lex_input = zopen(lex_file_name, 'rb')
	pos_input = zopen(pos_file_name, 'rb')
	tar_input = zopen(tar_file_name, 'rb')
	
	# input_files = [join(input_dir, f_) for f_ in listdir(input_dir) if isfile(join(input_dir, f_))]
	#
	# with open(pos_tok_file, "rb") as dict_file:
	# 	pos_tokeniser = pickle.load(dict_file)
	# with open(lex_tok_file, "rb") as dict_file:
	# 	lex_tokeniser = pickle.load(dict_file)
	
	learning_rate = CustomSchedule(hp_lex["dim"])
	opt_adam = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
	loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
	pred_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none')
	pred_loss_mean = tf.keras.metrics.Mean(name='prediction_train_loss')
	lex_train_loss_mean = tf.keras.metrics.Mean(name='lexical_train_loss')
	lex_perplexity = tf.keras.metrics.Mean(name='lexical_perplexity')
	lex_train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='lexical_train_accuracy')
	pos_train_loss_mean = tf.keras.metrics.Mean(name='pos_train_loss')
	pos_perplexity = tf.keras.metrics.Mean(name='pos_perplexity')
	pos_train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='pos_train_accuracy')
	pred_train_accuracy = tf.keras.metrics.Accuracy(name='pred_train_accuracy')
	metrics_history = {'lex loss': [], 'lex perp': [], 'lex acc': [], 'pos loss': [], 'pos perp': [], 'pos acc': [], 'pred loss': [], 'pred acc': []}
	
	def loss_function(real, pred):
		mask = tf.math.logical_not(tf.math.equal(real, 0))
		loss_ = loss_object(real, pred)
		perp_ = tf.exp(loss_)
		mask = tf.cast(mask, dtype=loss_.dtype)
		loss_ *= mask
		perp_ *= mask
		return tf.reduce_mean(loss_), tf.reduce_mean(perp_)
	
	form_two = architecture(hp_lex["layers"],
							hp_lex["dim"],
							hp_lex["pos_dim"],
							hp_lex["dff"],
							hp_lex["heads"],
							hp_lex["vocab"] + 4,  # start, end, unknown, special tokens
							hp_lex["pos_vocab"] + 4,
							rate=hp_lex["rate"],
							max_len=max_length)
	count_p = tf.keras.backend.count_params
	
	checkpoint_path = "./checkpoints/form2/" + chkpt_dir
	ckpt = tf.train.Checkpoint(transformer=form_two, optimizer=opt_adam)
	ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
	# if a checkpoint exists, restore the latest checkpoint.
	if ckpt_manager.latest_checkpoint:
		ckpt.restore(ckpt_manager.latest_checkpoint)
		print('Latest Form2 checkpoint restored!!')
	
	@tf.function
	def train_step(inputs_, targets_, trn_lbl, training=True):
		lex_targets_, pos_targets_, pred_targets_ = targets_
		pred_targets_ = tf.expand_dims(pred_targets_, axis=0)
		with tf.GradientTape(persistent=True) as tape:
			words_out, pos_out, pred_out = form_two(inputs_, turn_labels=trn_lbl, training=training)
			"""
			inputs: tuple( words, pos )
					words = list [target, history...]
					pos = list [target, history...]
			history -> most ancient ... most recent
			"""
			lex_loss, lex_perp = loss_function(lex_targets_, words_out)
			pos_loss, pos_perp = loss_function(pos_targets_, pos_out)
			pred_loss_val = pred_loss(pred_targets_, pred_out)
		trn_vars = form_two.trainable_variables
		pos_grads = tape.gradient(pos_loss, trn_vars)
		lex_grads = tape.gradient(lex_loss, trn_vars)
		pred_grads = tape.gradient(pred_loss_val, trn_vars)
		opt_adam.apply_gradients(zip(pos_grads, trn_vars))
		opt_adam.apply_gradients(zip(lex_grads, trn_vars))
		opt_adam.apply_gradients(zip(pred_grads, trn_vars))
		del tape
		lex_train_loss_mean(lex_loss)
		lex_perplexity(lex_perp)
		lex_train_accuracy(lex_targets_, words_out)
		pos_train_loss_mean(pos_loss)
		pos_perplexity(pos_perp)
		pos_train_accuracy(pos_targets_, pos_out)
		pred_loss_mean(pred_loss_val)
		pred_train_accuracy(pred_targets_, pred_out)
		return
	
	for epoch in range(epochs):
		print()
		lex_input.seek(0)
		pos_input.seek(0)
		tar_input.seek(0)
		lex_train_loss_mean.reset_states()
		lex_train_accuracy.reset_states()
		pos_train_loss_mean.reset_states()
		pos_train_accuracy.reset_states()
		start_ = time()
		step_count = 1
		for lex, pos, tar in zip(lex_input, pos_input, tar_input):
			"""
			inputs: tuple( words, pos )
					words = list [target, history...]
					pos = list [target, history...]
			history -> most ancient ... most recent
			targets = lex_targets, pos_targets, pred_targets
			"""
			lex = [str2tensor(context) for context in lex.decode(encoding).split(delimitter)]
			pos = [str2tensor(context) for context in pos.decode(encoding).split(delimitter)]
			tar = tar.decode(encoding).split(delimitter)
			turns = str2int(tar[0])
			targets = [tf.convert_to_tensor(str2int(item)) for item in tar[1:]]
			inputs = (lex, pos)
			train_step(inputs, targets, turns)
			if step_count % (batch_size * 50) == 0:
				print('Epoch {} Batch {}'.format(epoch + 1, int(step_count / batch_size)))
				print('Lexical: Loss {:.4f} Perplexity {:.4f} Accuracy {:.4f}'.format(lex_train_loss_mean.result(), lex_perplexity.result(), lex_train_accuracy.result()))
				print('POS: Loss {:.4f} Perplexity {:.4f} Accuracy {:.4f}'.format(pos_train_loss_mean.result(), pos_perplexity.result(), pos_train_accuracy.result()))
				print('Label: Loss {:.4f} Accuracy {:.4f}'.format(pred_loss_mean.result(), pred_train_accuracy.result()))
			step_count += 1
		trainable = None
		if not trainable:
			trainable = np.sum([count_p(p) for p in set(form_two.trainable_weights)])
		print()
		print('Epoch {}'.format(epoch + 1))
		print('time taken for epoch: {} secs'.format(time() - start_))
		ckpt_save_path = ckpt_manager.save()
		print('saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))
		print('total trainable variables: {}'.format(trainable))
		# print('sample results')
		# print('lexical: ' + str(tf.argmax(res_lex, axis=-1).numpy()))
		# print('pos: ' + str(tf.argmax(res_pos, axis=-1).numpy()))
		# print('label: ' + str(res_pred.numpy()))
		print('Lexical: Loss {:.4f} Perplexity {:.4f} Accuracy {:.4f}'.format(lex_train_loss_mean.result(), lex_perplexity.result(), lex_train_accuracy.result()))
		metrics_history['lex loss'].append(float(lex_train_loss_mean.result()))
		metrics_history['lex perp'].append(float(lex_perplexity.result()))
		metrics_history['lex acc'].append(float(lex_train_accuracy.result()))
		print('POS: Loss {:.4f} Perplexity {:.4f} Accuracy {:.4f}'.format(pos_train_loss_mean.result(), pos_perplexity.result(), pos_train_accuracy.result()))
		metrics_history['pos loss'].append(float(pos_train_loss_mean.result()))
		metrics_history['pos perp'].append(float(pos_perplexity.result()))
		metrics_history['pos acc'].append(float(pos_train_accuracy.result()))
		print('Label: Loss {:.4f} Accuracy {:.4f}'.format(pred_loss_mean.result(), pred_train_accuracy.result()))
		metrics_history['pred loss'].append(float(pred_loss_mean.result()))
		metrics_history['pred acc'].append(float(pred_train_accuracy.result()))
		print()
	for key in metrics_history:
		max_ = max(metrics_history[key])
		if max_ == 0:
			max_ = 1
		metrics_history[key] = [float(i)/max_ for i in metrics_history[key]]
	with open(chkpt_dir + ' metrics history.csv', 'w') as csv_file:
		wr = csv.writer(csv_file)
		keys = list(metrics_history.keys())
		wr.writerows([keys])
		wr.writerows(zip(*[metrics_history[key] for key in keys]))


def test():
	from nltk.corpus import switchboard
	from random import randint
	lex_tok = tf.keras.preprocessing.text.Tokenizer(num_words=100, filters='\t\n', split=' ', oov_token='___', lower=True)
	pos_tok = tf.keras.preprocessing.text.Tokenizer(num_words=30, filters='\t\n', split=' ', oov_token='___', lower=True)
	texts = [list(turn) for turn in switchboard.tagged_turns()]
	lex_ = []
	pos_ = []
	for turn in texts:
		lex_.append(" ".join([l for l, p in turn]))
		pos_.append(" ".join([p for l, p in turn]))
	del texts
	lex_tok.fit_on_texts(lex_)
	pos_tok.fit_on_texts(pos_)
	form_two = FormTwo.FormTwo(1, 32, 32, 32, 4, 100, 30, 0.1)
	count_p = tf.keras.backend.count_params
	sent_len = 20
	for counter in range(10):
		print()
		index = randint(0, len(lex_))
		lex = lex_tok.texts_to_sequences(lex_[index:index + 1])
		lex = lex[0]
		lex = list(truncate_split(lex, sent_len-3))
		print(lex)
		pos = pos_tok.texts_to_sequences(pos_[index:index + 1])
		pos = pos[0]
		pos = list(truncate_split(pos, sent_len-3))
		print(pos)
		if lex is None or pos is None:
			continue
		lex = [pad2len(tf.convert_to_tensor([_lex]), sent_len) for _lex in lex]
		pos = [pad2len(tf.convert_to_tensor([_pos]), sent_len) for _pos in pos]
		tar_pos = pos[-1]
		tar_lex = lex[-1]
		# lex = "The quick, brown fox jumps over the lazy dog."
		# pos = "DET ADJ . ADJ NOUN VERB ADP DET ADJ NOUN ."
		inputs = ([tar_lex] + lex, [tar_pos] + pos)
		print("inputs:")
		print(inputs)
		turn_labels = [1 if (i % 2 == 0) else 0 for i in range(len(inputs[0]))]
		res_lex, res_pos, res_pred = form_two(inputs, turn_labels, training=True)
		dummy = form_two.get_trainable_vars()
		del dummy
		print("lex")
		print("target")
		print(tar_lex.shape)
		print("result")
		print(res_lex.shape)
		out_ = tf.argmax(res_lex, axis=-1).numpy()
		print(out_)
		print(lex_tok.sequences_to_texts(out_))
		print("pos")
		print("target")
		print(tar_pos.shape)
		print("result")
		print(res_pos.shape)
		out_ = tf.argmax(res_pos, axis=-1).numpy()
		print(out_)
		print(pos_tok.sequences_to_texts(out_))
		print("pred")
		print("{} , {}".format(len(inputs[0]), len(inputs[1])))
		print(res_pred.shape)
		print(res_pred.numpy())
		print()
	trainable = np.sum([count_p(p) for p in set(form_two.trainable_weights)])
	print(trainable)
	return


if __name__ == '__main__':
	# test()
	main()
	# ToDo: evaluate
	# ToDo: experiment with various loss functions
	
	# ToDo: implement Form One
	# ToDo: sub-word tokeniser ?
