# Author: Jie Zhao
# This implements the review-based qa for binary questions, where we
# also simultaneously learn the relevance scores.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import sys
import os
import numpy as np
import pdb
import tensorflow as tf
from configs import NUMERIC_CHECK, NUMERIC_CHECK_ALL

from model_rnn import POOLING

from eval_metrics import EvalStats

idx2category = {0: "True positive",
								1: "True negative",
								2: "False positive",
								3: "False negative"}

def show_qual_rslt(mydata,
	question,
	sentences,
	label,
	preds, # [batch_size]
	costs, # [batch_size]
	sent_labels, # [batch_size, sent_num]
	sent_preds,	# [batch_size, sent_num]
	sent_costs,	# [batch_size, sent_num]
	regus=None,	# [batch_size]
	ques_attn=None, # [batch_size, ques_len]
	sent_attn=None, # [batch_size, sent_num, sent_len]
	pw_sent_preds=None,
	pw_preds=None,
	s_w_mask=None,
	s_mask=None,
	rslt=None,
	batch_size=1,
	):
	""" Display qualitative results.
	"""
	question_length = len(question[0])
	sentence_number = len(sentences[0])
	sentence_length = len(sentences[0, 0])

	# pdb.set_trace()
	for batch_id in xrange(batch_size):
		# DISPLAY QUESTIONS
		print('-----\nQuestions:')
		if POOLING == 'attention':
			assert(len(ques_attn[batch_id]) == question_length)
		tokens, attens = [], []
		printtoscreen = ''
		for idx in xrange(question_length):
			token = question[batch_id,idx]
			if POOLING != 'attention': atten = 0
			else: atten = ques_attn[batch_id, idx]
			if token == 0: assert(atten < 1e-8)
			else:
				if POOLING == 'attention':
					printtoscreen += ' ({}, {:.3f})'.format(mydata.id2word[token], atten)
				else:
					printtoscreen += ' {}'.format(mydata.id2word[token])
		print(printtoscreen)

		# DISPLAY SENTENCES
		print('Sentences:\n%-5s %-5s %-5s %-5s' %
					('ID', 'LABEL', 'PRED', 'COST'))
		for i in xrange(sentence_number):
			if s_mask is not None and s_mask[batch_id][i] == 0:
				continue # Skip all padding sentences
			# print('%-5s %-5d %-5.3f %-5.3f: ' % (('-%d-'%i), sent_labels[batch_id,i],
			#				sent_preds[batch_id,i], sent_costs[batch_id,i]), end='')
			print('%-5s %-5d %-5.3f %-5.3f: ' % (('-%d-'%i), sent_labels[batch_id,i],
						sent_preds[batch_id,i], -1), end='')
			tokens, attens = [], []
			printtoscreen = ''
			for idx in xrange(sentence_length):
				token = sentences[batch_id, i, idx]
				if POOLING != 'attetion': atten = 0
				else: atten = sent_attn[batch_id, i, idx]
				if token == 0: assert(atten < 1e-8)
				else:
					if POOLING == 'attention':
						printtoscreen += ' ({} {:.3f})'.format(mydata.id2word[token], atten)
					else:
						# printtoscreen += ' {}<{:d}>'.format(mydata.id2word[token], idx)
						printtoscreen += ' {}'.format(mydata.id2word[token])
			print(printtoscreen)

			# DISPLAY PAIRWISE PREDICTION RESULTS
			printtoscreen = '\n'
			if pw_sent_preds is not None:
				pw_preds = pw_sent_preds[batch_id,:,:,i]
				# Tabel header
				printtoscreen += '{:>15} | '.format('')
				for s_step in xrange(pw_preds.shape[1]):
					if sentences[batch_id,i,s_step] == 0: continue # Skip if sentence token is padding
					printtoscreen += '{:5s} '.format('<{:d}>'.format(s_step))
				printtoscreen += '\n'
				# Table body
				for q_step in xrange(pw_preds.shape[0]):
					q_w_id = question[batch_id,q_step]
					if q_w_id == 0: continue # Skip if question token is padding
					printtoscreen += '{:>15} | '.format(mydata.id2word[q_w_id][:15])
					for s_step in xrange(pw_preds.shape[1]):
						if sentences[batch_id,i,s_step] == 0: continue # Skip if sentence token is padding
						printtoscreen += '{:.3f} '.format(pw_preds[q_step, s_step])
					printtoscreen += '\n'
				print(printtoscreen)
				# print(np.array_str(pw_sent_preds[:,:,:,i], precision=3, max_line_width=400))

		if rslt is not None: print(idx2category[rslt[batch_id]])
		if regus is None:
			print('Answer trigger: truth=%d, prediction=%.3f, cost=%.3f (batch)' %
						(label[batch_id], preds[batch_id], costs))
		else:
			print('Answer trigger: truth=%d, prediction=%.3f, cost=%.3f (batch), regularization=%.3f (batch)' %
						(label[batch_id], preds[batch_id], costs, regus))


def test_model(FLAGS, session, m, config, mydata, if_test=False,
							 if_show_qual_rslt=False, if_load_ckpt=False):
	""" Run validation (if_test = False) OR test (if_test=True).
	"""
	start_time = time.time()
	costs = []
	cost_list = []
	regus = []

	if if_test:
		print ("----- ----- %sTesting ----- -----") 
		buckets = mydata.test_buckets
	else:
		print ("----- ----- %sValidation ----- -----")
		buckets = mydata.valid_buckets
	if if_load_ckpt:
		print ("----- ----- %sLoading checkpoint ----- -----")
		restore_checkpoint(FLAGS, session, m)

	m.eval_stat.reset()
	for bucket_id, bucket in enumerate(buckets):
		sentence_number, question_length, sentence_length = bucket
		# for question, sentences, label, sent_labels, q_mask, s_w_mask, s_mask in \
		#			mydata.get_test(bucket_id, if_test=if_test):
		for question, sentences, label, sent_labels, q_mask, s_w_mask, s_mask, q_feat, s_feat, \
				overlap in mydata.get_test_batch(bucket_id,
																				 m.batch_size,
																				 if_test=if_test):
			# Dimension check
			assert(len(question[0]) == question_length)
			for i in xrange(sentence_number):
				assert(len(sentences[0, i]) == sentence_length)

			# Feed data
			feed_dict = {}
			feed_dict[m.input_question_feat.name] = q_feat
			for i in xrange(question_length):
				feed_dict[m.input_question[i].name] = question[:, i]
				feed_dict[m.input_question_word_mask[i].name] = q_mask[:, i]
			for i in xrange(sentence_number):
				feed_dict[m.input_sentence_feat[i].name] = s_feat[:, i]
				feed_dict[m.input_sentence_mask[i].name] = s_mask[:, i]
				feed_dict[m.sent_targets[i].name] = sent_labels[:, i]
				for j in xrange(sentence_length):
					feed_dict[m.input_sentences[i][j].name] = sentences[:, i, j]
					feed_dict[m.input_sentence_word_mask[i][j].name] = s_w_mask[:, i, j]

			feed_dict[m.targets.name] = label
			# Heterogeneous label mask
			targets_avg = np.sum(sent_labels * s_mask, axis=1, keepdims=True) / \
										np.sum(s_mask, axis=1, keepdims=True) # [batch_size]T
			targets_mask = (np.sum(np.abs(sent_labels - targets_avg) * s_mask,
														 axis=1) > 0).astype(np.float32) # [batch_size]
			feed_dict[m.targets_mask.name] = targets_mask

			# Fetch results
			fetches = [m.cost_b[bucket_id],
								 m.sent_pred_b[bucket_id],
								 m.pred_b[bucket_id],
								 m.regu_b[bucket_id],
								 m.question_attention_b[bucket_id],
								 m.sentence_attention_b[bucket_id],
								 m.sent_cost_b[bucket_id],
								 m.cost_list_b[bucket_id]]
			fetch_results = session.run(fetches, feed_dict)

			costs.append(fetch_results[0]) # []
			sent_preds = fetch_results[1]
			preds = fetch_results[2]
			regus.append(fetch_results[3])
			ques_attn = fetch_results[4] # [bathc_size, ques_len]
			sent_attn = fetch_results[5] # [batch_size, sent_num, sent_len]
			sent_costs = fetch_results[6] # [batch_size, sent_num]
			cost_list.append(fetch_results[7]) # [3]

			# print("OVERLAP=%d" % overlap)
			assert(len(sent_labels) == m.batch_size)
			assert(len(sent_preds) == m.batch_size)
			assert(len(s_mask) == m.batch_size)
			rslt = m.eval_stat.update_stats(
								 s_label=sent_labels[overlap:], # sentence level label
								 s_preds=sent_preds[overlap:],	# sentence level prediction
								 mask=s_mask[overlap:])

			if if_show_qual_rslt:
				# SHOW SOME QUALITATIV
				# NOTE: assume batch size is 1, so the first index of 'question' and
				#				'sentences' is always 0.E RESULTS
				show_qual_rslt(mydata, question[overlap:], sentences[overlap:],
											 label[overlap:], preds[overlap:], costs[-1],
											 sent_labels[overlap:], sent_preds[overlap:], sent_costs,
											 regus=regus[-1],
											 ques_attn=ques_attn, sent_attn=sent_attn,
											 rslt=rslt, pw_sent_preds=pw_sent_preds,
											 s_w_mask=s_w_mask, s_mask=s_mask,
											 batch_size=m.batch_size - overlap)

	# Stats
	cost = np.mean(costs)
	cost1, cost2, cost3 = np.mean(np.array(cost_list), axis=0)
	regu = np.mean(regus)
	accuracy, bag_accuracy = m.eval_stat.accuracy()
	precision, bag_precision = m.eval_stat.precision()
	recall, bag_recall = m.eval_stat.recall()
	f1, bag_f1 = m.eval_stat.f1()
	MAP = m.eval_stat.MAP()
	MRR = m.eval_stat.MRR()
	tag = 'test' if if_test else 'valid'
	print("- %s costs     : %.5f (regu: %.5f)" % (tag, cost, regu))
	print("- %s cost list : %.5f %.5f %.5f" % (tag, cost1, cost2, cost3))
	print("- %s accuracy  : %.5f (%.5f)" % (tag, accuracy, bag_accuracy))
	print("- %s precision : %.5f (%.5f)" % (tag, precision, bag_precision))
	print("- %s recall    : %.5f (%.5f)" % (tag, recall, bag_recall))
	print("- %s f1 score  : %.5f (%.5f)" % (tag, f1, bag_f1))
	print("- %s MAP	      : %.5f" % (tag, MAP))
	print("- %s MRR	      : %.5f" % (tag, MRR))
	print("Time %.0f seconds" % (time.time() - start_time))

	return cost, accuracy, precision, recall, f1


def train_model(FLAGS, session, m, config, mydata, m_valid=None):
	""" Run training steps.
	NOTE: the buckets here is NOT the collapsed buckets
	"""

	if FLAGS.continue_training:
		print('----- Loading checkpoints -----')
		restore_checkpoint(FLAGS, session, m)
	else:
		print('----- Using fresh parameters -----')
		session.run(tf.initialize_all_variables())
		# session.run(tf.global_variables_initializer())
	print('----- Initialization finished -----')

	# Check initialized embedding matrix is as expected
	embed_matrix = session.run(m.embedding)
	assert((np.abs(embed_matrix - mydata.word_embeddings) < 1e-8).all())

	# SAMPLEBAG
	num_step_in_epoch = sum(mydata.train_bucket_sizes) // config.batch_size + 1
	# num_step_in_epoch = np.sum(mydata.train_all_sent_num_b) // (5 * config.batch_size) + 1

	min_valid_cost, max_valid_accu, max_valid_f1 = None, None, None
	valid_costs, valid_accus, valid_f1s = [], [], []
	if_saved_to_checkpoint = False
	min_test_cost, max_test_accu, max_test_f1 = None, None, None
	test_costs, test_accus, test_f1s = [], [], []

	for epoch_cnt in range(config.max_max_epoch):
		print("\n----- ----- %sTraining ----- -----")
		# if	epoch_cnt > config.max_epoch: session.run(m._lr_update)
		# print("Epoch: %d \tLearning rate: %.3f" % (epoch_cnt, m.lr.eval()))
		print("Epoch: %d" % (epoch_cnt))
		m.eval_stat.reset()
		costs, regus = 0.0, 0.0
		cost_list = []
		start_time = time.time()
		for step in xrange(num_step_in_epoch):
			# print(m.global_step.eval())
			random_number_01 = np.random.random_sample()
			bid = mydata.sample_train_buckets()
			cost, regu, cost_parts = \
					train_step(session, m, mydata, bid, m.batch_size)
			costs += cost
			regus += regu
			cost_list.append(cost_parts)
		avg_cost = float(costs) / num_step_in_epoch
		avg_regu = float(regus) / num_step_in_epoch
		cost1, cost2, cost3 = np.mean(np.array(cost_list), axis=0)

		# print("- cost : %.5f (batch) %.5f (single)" %
		#				(avg_cost, avg_cost / m.batch_size))
		# print("- cost list : %.5f %.5f %.5f (batch)" %
		#				(cost1, cost2, cost3))
		# print("- regu : %.5f (batch) %.5f (single)" %
		#				(avg_regu, avg_regu / m.batch_size))
		print("- cost : %.5f" % (avg_cost))
		print("- cost list : %.5f %.5f %.5f" %
					(cost1, cost2, cost3))
		print("- regu : %.5f" % (avg_regu))
		print("- accuracy  : %.5f (%.5f)" % m.eval_stat.accuracy())
		print("- precision : %.5f (%.5f)" % m.eval_stat.precision())
		print("- recall		 : %.5f (%.5f)" % m.eval_stat.recall())
		print("- f1				 : %.5f (%.5f)" % m.eval_stat.f1())
		print("- MAP : %.5f" % m.eval_stat.MAP())
		print("- MRR : %.5f" % m.eval_stat.MRR())
		print("Time %.0f seconds" % (time.time() - start_time))

		# VALIDATION
		if m_valid:
			# FIXME: using training configuration actually
			# But 'config' is not used anywhere in 'test_model'
			cost, accuracy, precision, recall, f1 = \
					test_model(FLAGS, session, m_valid, config, mydata,
										if_test=False)
			valid_costs.append(cost)
			valid_accus.append(accuracy)
			valid_f1s.append(f1)
			# Save checkpoint and zero timer and loss.
			if (epoch_cnt % FLAGS.epochs_per_validation == 0):
				valid_cost = np.mean(valid_costs)
				valid_accu = np.mean(valid_accus)
				valid_f1	 = np.mean(valid_f1s)
				del valid_costs[:], valid_accus[:], valid_f1s[:]
				if min_valid_cost is None: min_valid_cost = valid_cost
				elif min_valid_cost > valid_cost:
					min_valid_cost = valid_cost
				if max_valid_accu is None: max_valid_accu = valid_accu
				elif max_valid_accu < valid_accu:
					max_valid_accu = valid_accu
				if max_valid_f1 is None: max_valid_f1 = valid_f1
				elif max_valid_f1 < valid_f1:
					max_valid_f1 = valid_f1
					if_saved_to_checkpoint = True
					save_checkpoint(FLAGS, session, m)
				print("Valid cost min: %5f" % (min_valid_cost
																			 if min_valid_cost is not None else -1.0))
				print("Valid accu max: %5f" % (max_valid_accu
																			 if max_valid_accu is not None else -1.0))
				print("Valid f1 max  : %5f" % (max_valid_f1
																			 if max_valid_f1 is not None else -1.0))
			if (epoch_cnt == config.max_max_epoch-1 and not if_saved_to_checkpoint):
				print("Nothing saved !!!")
				# save_checkpoint(FLAGS, session, m)

  if m_valid:
    cost, accuracy, precision, recall, f1 = \
        test_model(FLAGS, session, m_valid, config, mydata,
                  if_test=True, if_pretrain=if_pretrain)
      print("test cost : %5f" % (cost))
      print("test accu : %5f" % (accuracy))
      print("test f1 :   %5f" % (f1))




def train_step(session, model, mydata, bucket_id, batch_size):
	""" Run a single batch of data. Optionally update the trainables.
	Used for training and validation.
	"""
	question, sentences, label, sent_label, q_mask, s_w_mask, s_mask, q_feat, s_feat = \
			mydata.get_train_batch(bucket_id, batch_size)
	# SAMPLEBAG
	sentence_number = mydata.train_buckets[bucket_id][0]
	# sentence_number = 5

	question_length = mydata.train_buckets[bucket_id][-2]
	sentence_length = mydata.train_buckets[bucket_id][-1]
	assert(question.shape == (batch_size, question_length))
	assert(sentences.shape == (batch_size, sentence_number, sentence_length))
	assert(label.shape == (batch_size,))

	# Feed data
	feed_dict = {}
	feed_dict[model.input_question_feat.name] =	q_feat
	for i in xrange(question_length):
		feed_dict[model.input_question[i].name] = question[:, i]
		feed_dict[model.input_question_word_mask[i].name] = q_mask[:, i]
	for i in xrange(sentence_number):
		feed_dict[model.input_sentence_feat[i].name] = s_feat[:, i]
		feed_dict[model.input_sentence_mask[i].name] = s_mask[:, i]
		feed_dict[model.sent_targets[i].name] = sent_label[:, i]
		for j in xrange(sentence_length):
			feed_dict[model.input_sentences[i][j].name] = sentences[:, i, j]
			feed_dict[model.input_sentence_word_mask[i][j].name] = s_w_mask[:, i, j]

	feed_dict[model.targets.name] = label
	# Heterogeneous label mask
	targets_avg = np.sum(sent_label * s_mask, axis=1, keepdims=True) / \
								np.sum(s_mask, axis=1, keepdims=True) # [batch_size]T
	targets_mask = (np.sum(np.abs(sent_label - targets_avg) * s_mask,
												 axis=1) > 0).astype(np.float32) # [batch_size]
	feed_dict[model.targets_mask.name] = targets_mask
	# Fetch results
	fetches = [model.cost_b[bucket_id],
						 model.pred_b[bucket_id],
						 model.regu_b[bucket_id],
						 model.sent_pred_b[bucket_id],
						 model.cost_list_b[bucket_id],
						 model.train_op_b[bucket_id]]

	fetch_results = session.run(fetches, feed_dict)
	cost = fetch_results[0]
	preds = fetch_results[1]
	regu = fetch_results[2]
	s_preds = fetch_results[3]
	cost_list = fetch_results[4]

	try:
		assert((preds == np.max(s_preds * s_mask, axis=1)).all())
	except AssertionError as e:
		print(e)
		print(preds)
		print(s_preds)
		raise(e)
	model.eval_stat.update_stats(s_label=sent_label, # sentence level label
															 s_preds=s_preds, # sentence level prediction
															 mask=s_mask)
	return cost, regu, cost_list


def restore_checkpoint(FLAGS, session, m):
	start_time = time.time()
	checkpoint_path = os.path.join(FLAGS.ckpt_dir, "MIL_AnswerTrigger.ckpt")
	ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_dir + FLAGS.ckpt_suffix)
	# assert(ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path))
	assert(ckpt)
	m.saver.restore(session, ckpt.model_checkpoint_path)
	print("-----\nCheckpoint restored from '%s'. (%f sec)" %
				(checkpoint_path, time.time() - start_time))

def save_checkpoint(FLAGS, session, m):
	start_time = time.time()
	checkpoint_path = os.path.join(FLAGS.ckpt_dir + FLAGS.ckpt_suffix,
				       "MIL_AnswerTrigger.ckpt")
	if not os.path.exists(checkpoint_path):
		os.makedirs(checkpoint_path)
	save_path = m.saver.save(session, checkpoint_path, global_step=m.global_step,
				 write_meta_graph=False)
	print("-----\nCheckpoint saved to '%s'. (%f sec)" %
				(save_path, time.time() - start_time))

