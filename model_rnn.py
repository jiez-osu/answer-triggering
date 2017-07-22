# Copyright: Jie Zhao
# This file contains functions for tensorflow computation graph construction.
# Most likely these functiosn will take as input somoe tensors and return
# some tensors as well.
#===============================================================================

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell, nn_ops
import pdb
import sys
from configs import NUMERIC_CHECK
import numpy as np

POOLING='average' # Can be 'average', 'max' or 'attention'

#===============================================================================
# Below are helper functions that can help build the model
#===============================================================================

def qa_rnn(config, is_training, input_seqs, input_masks, xvector=None):
	"""Model that takes as input several input sequences, output the encoded vector
	for each of the sequence.
	Args:
		is_training: boolean. Indictates if the model is used for training or not.
		input_seq: 2-D list of tensors, each is a [batch_size * embed_size] tensor.
	"""
	embed_seqs, atten_seqs, all_output_seqs = \
			sentences_encoding(config, is_training, input_seqs, input_masks, xvector)
	if NUMERIC_CHECK:
		embed_seqs = tf.check_numerics(embed_seqs, 'qa_rnn output embedding numeric error')
		atten_seqs = tf.check_numerics(atten_seqs, 'qa_rnn output attention numeric error')
		all_output_seqs = tf.check_numerics(all_output_seqs, 'qa_rnn output numeric error')
	return embed_seqs, atten_seqs, all_output_seqs


def sentences_encoding(config, is_training,
											 input_seqs,	# list of list of tensors: 1st idx is sent, 2nd is token
											 input_masks, # list of list of tensors: 1st idx is sent, 2nd is token
											 xvector=None):
	""" Embed input_seqs into a vector space using an attention LSTM model.
	"""
	num_sent = len(input_seqs)
	with tf.variable_scope("cell"):

		# lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(config.rnn_hidden_size,
		#																					 forget_bias=0.0)
		lstm_cell = tf.nn.rnn_cell.GRUCell(config.rnn_hidden_size)

		if is_training and config.rnn_keep_prob < 1:
			lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
					lstm_cell,
					input_keep_prob=1.0,
					output_keep_prob=config.rnn_keep_prob)

		if config.num_layers > 1:
			cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)
		else:
			cell = lstm_cell

	embed_seqs = []
	atten_seqs = []
	all_output_seqs = []
	for i in xrange(num_sent):
		if i > 0: tf.get_variable_scope().reuse_variables()
		if xvector is None: # Q AND S CALCULATION THEIR ATTENTION SEPARATELY
			weighted_sum, outputs_concat, hidden_outputs, attention_weights = \
					attention_rnn(config, cell,
												input_seqs[i],	# list of [batch_size, embed_size]
												input_masks[i], # list of [batch_size]
												pooling=POOLING)
		else: # CROSS ATTENTION: Q VECTOR INFLUENCE S ATTENTION WEIGHTS
			weighted_sum, outputs_concat, hidden_outputs, attention_weights = \
					cross_attention_rnn(config, cell,
															input_seqs[i],
															input_masks[i],
															xvector)
		embed_seqs.append(weighted_sum)
		atten_seqs.append(attention_weights)
		all_output_seqs.append(outputs_concat)
	# pdb.set_trace()
	packed_embed_seqs = tf.pack(embed_seqs, 1) # [batch_size, sent_num, hidden_size]
	packed_atten_seqs = tf.pack(atten_seqs, 1) # [batch_size, sent_num, num_step]
	packed_all_output_seqs = tf.pack(all_output_seqs, 1) # [batch_size, sent_num, num_step, hidden_size]

	return packed_embed_seqs, packed_atten_seqs, packed_all_output_seqs


def attention_rnn(config, cell,
									inputs,				# list of [batch_size, embed_size]
									padding_mask, # list of [batch_size]
									pooling='max'):
	""" Input a list of tensors and get back the embedded vector for this list.
	"""
	num_steps = len(inputs)
	assert(len(inputs) == len(padding_mask))
	hidden_size = cell.output_size * 2
	batch_size = inputs[0].get_shape()[0].value
	embed_size = inputs[0].get_shape()[1].value
	assert(cell.output_size == config.rnn_hidden_size)
	assert(batch_size == config.batch_size)
	assert(embed_size == config.word_embed_size)

	with tf.variable_scope("attention_RNN"):
		input_length = tf.reduce_sum(tf.pack(padding_mask, axis=1), 1)
		# input_length = tf.Print(input_length,
		#													[padding_mask, input_length, inputs],
		#													message='input length', summarize=100)
		outputs, state_fw, state_bw = \
				tf.nn.bidirectional_rnn(cell, cell,
																inputs, # list of [batch_size, embed_size]
																dtype=config.data_type,
																sequence_length=input_length)

		# pdb.set_trace()

		# CHECK
		zero_check_ops = []
		ones = tf.constant(1, shape=[batch_size], dtype=config.data_type)
		zeros = tf.constant(0, shape=[batch_size, hidden_size], dtype=config.data_type)
		for i in xrange(num_steps):
			reverse_padding_mask = tf.abs(ones - padding_mask[i])
			#zero_check_ops.append(tf.assert_equal(
			#		 outputs[i] * tf.expand_dims(reverse_padding_mask, 1), zeros,
			#		 data=[padding_mask[i], reverse_padding_mask, outputs[i]], summarize=500,
			#		 message='Bidirectional RNN output error'))

		# RESHAPE THE OUTPUTS, JUST IN CASE NONE DIM
		with tf.control_dependencies(zero_check_ops):
			shaped_outputs = [tf.reshape(o, [batch_size, hidden_size])
												for o in outputs]
			outputs = shaped_outputs

		# OVERALL SEQUENCE REPRESENTAION
		hidden_outputs = []
		attention_weights = []
		outputs_concat = tf.pack(outputs, axis=1) # [batch_size, num_step, hidden_size]
		if pooling == 'attention': # USING ATTENTION MECHANISM
			with tf.variable_scope("attention_computation"):
				context_vector = tf.get_variable("context_vector", [hidden_size, 1])
				# Calculate attention
				for i in xrange(len(outputs)):
					if i > 0: tf.get_variable_scope().reuse_variables()
					hidden_output = tf.tanh(rnn_cell._linear(outputs[i], hidden_size,
																									 True # If add bias
																									 ))
					hidden_outputs.append(hidden_output)
					attention_weights.append(tf.matmul(hidden_output, context_vector)) # [batch_size, 1]
				attention_weights = tf.concat(1, attention_weights)
				attention_weights = tf.nn.softmax(attention_weights) * \
														tf.pack(padding_mask, axis=1) # [batch_size, num_steps]
				attention_weights = tf.div(attention_weights,
																	 1e-12 + tf.reduce_sum(attention_weights, 1, keep_dims=True)) # [batch_size, num_steps]
				# Attention weighted sum
				weighted_sum = tf.reduce_sum(outputs_concat * tf.expand_dims(attention_weights, 2),
																		 1) # [batch_size, hidden_size]
		elif pooling == 'average': # AVERAGE POOLING
			weighted_sum = tf.reduce_mean(outputs_concat, 1) # [batch_size, hidden_size]
		elif pooling == 'max': # Max pooling
			weighted_sum = tf.reduce_max(outputs_concat, 1)  # [batch_size, hidden_size]
		else:
			raise ValueError("Unknown pooling method: '{}'".format(pooling))

	return weighted_sum, outputs_concat, hidden_outputs, attention_weights



def cross_attention_rnn(config, cell,
												inputs,
												padding_mask,
												xvector):
	""" Input a list of tensors and get back the embedded vector for this list.

	NOTE: the difference from this function to the above one is that this takes
				vector from another source into consideration when calculating attention
				weights. See Tan et al., 2015 "Lstm-based deep learning models for
				non-factoid answer selection" for details.
	"""
	num_steps = len(inputs)
	hidden_size = cell.output_size * 2
	batch_size = inputs[0].get_shape()[0].value
	embed_size = inputs[0].get_shape()[1].value
	assert(cell.output_size == config.rnn_hidden_size)
	assert(batch_size == config.batch_size)
	assert(embed_size == config.word_embed_size)

	with tf.variable_scope("attention_RNN"):
		input_length = tf.reduce_sum(tf.pack(padding_mask, axis=1), 1)
		# input_length = tf.Print(input_length, [padding_mask, input_length],
		#													message='input length', summarize=50)
		outputs, state_fw, state_bw = \
				tf.nn.bidirectional_rnn(cell, cell, inputs, dtype=config.data_type,
																sequence_length=input_length)

		# RESHAPE THE OUTPUTS, JUST IN CASE NONE DIM
		shaped_outputs = [tf.reshape(o, [batch_size, hidden_size]) for o in outputs]
		outputs = shaped_outputs
		outputs_for_attention = [tf.concat(1, [o, xvector]) # [batch_size, 2*hidden_size]
														 for o in outputs]

		# OVERALL SEQUENCE REPRESENTAION
		hidden_outputs = []
		attention_weights = []
		outputs_concat = tf.pack(outputs, axis=1) # [batch_size, num_step, hidden_size]
		with tf.variable_scope("attention_computation"):
			context_vector = tf.get_variable("context_vector", [2*hidden_size, 1])
			# Calculate attention
			attention_weights = []
			for i in xrange(len(outputs)):
				if i > 0: tf.get_variable_scope().reuse_variables()
				hidden_output = tf.tanh(rnn_cell._linear(outputs_for_attention[i],
																								 2*hidden_size,
																								 True # If add bias
																								 ))
				hidden_outputs.append(hidden_output)
				attention_weights.append(tf.matmul(hidden_output, context_vector)) # [batch_size, 1]
			attention_weights = tf.concat(1, attention_weights)
			attention_weights = tf.nn.softmax(attention_weights) * \
													tf.pack(padding_mask, axis=1) # [batch_size, num_steps]
			attention_weights = tf.div(attention_weights,
																 1e-12 + tf.reduce_sum(attention_weights, 1, keep_dims=True))
			# Attention weighted sum
			weighted_sum = tf.reduce_sum(outputs_concat * tf.expand_dims(attention_weights, 2),
																	 1) # [batch_size, hidden_size]

	return weighted_sum, outputs_concat, hidden_outputs, attention_weights
