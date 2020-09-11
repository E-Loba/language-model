import FormTwo
import numpy as np
from z_util import *
from gzip import open as zopen


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
	# ToDo: fit tokenisers to texts
	
	learning_rate = CustomSchedule(hp_lex["dim"])
	opt_adam = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
	pred_loss_proxy = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none', label_smoothing=0.1)  # ToDo: find appropriate prediction loss
	pred_train_loss_mean = tf.keras.metrics.Mean(name='pred_train_loss')
	pred_train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='pred_train_accuracy')
	metrics_history = {'pred loss': [], 'pred acc': []}
	
	form_two = architecture(hp_lex["layers"],
							hp_lex["dim"],
							hp_lex["pos_dim"],
							hp_lex["dff"],
							hp_lex["heads"],
							hp_lex["vocab"] + 4,  # start, end, unknown tokens
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
		with tf.GradientTape() as tape:
			words_out, pos_out, pred_out = form_two(inputs_, turn_labels=trn_lbl, training=training)
			"""
			inputs: tuple( words, pos )
					words = list [target, history...]
					pos = list [target, history...]
			history -> most ancient ... most recent
			"""
			pred_loss = pred_loss_proxy(pred_targets_, pred_out)
		trn_vars = form_two.trainable_variables
		pred_grad = tape.gradient(pred_loss, trn_vars)
		opt_adam.apply_gradients(zip(pred_grad, trn_vars))
		pred_train_loss_mean(pred_loss)
		pred_train_accuracy(pred_targets_, pred_out)  # ToDo: categorical labels
		return
	
	for epoch in range(epochs):
		print()
		lex_input.seek(0)
		pos_input.seek(0)
		tar_input.seek(0)
		pred_train_loss_mean.reset_states()
		pred_train_accuracy.reset_states()
		start_ = time()
		step_count = 1
		for lex, pos, tar in zip(lex_input, pos_input, tar_input):  # ToDo: this
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
				print('Feature: Loss {:.4f} Accuracy {:.4f}'.format(pred_train_loss_mean.result(), pred_train_accuracy.result()))
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
		print('Feature: Loss {:.4f} Accuracy {:.4f}\n'.format(pred_train_loss_mean.result(), pred_train_accuracy.result()))
		metrics_history['pred loss'].append(float(pred_train_loss_mean.result()))
		metrics_history['pred acc'].append(float(pred_train_accuracy.result()))
		print()
	for key in metrics_history:
		max_ = max(metrics_history[key])
		if max_ == 0:
			max_ = 1
		metrics_history[key] = [float(i) / max_ for i in metrics_history[key]]
	with open(chkpt_dir + ' binary metrics history.csv', 'w') as csv_file:
		wr = csv.writer(csv_file)
		wr.writerows(zip(*[metrics_history[key] for key in metrics_history]))


if __name__ == '__main__':
	main()
