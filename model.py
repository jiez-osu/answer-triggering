# Copyright Jie Zhao
#===============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import sys
import os
import pdb
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

from model_rnn import qa_rnn, POOLING
from model_group import group_learning
from configs import NUMERIC_CHECK, NUMERIC_CHECK_ALL
from eval_metrics import EvalStats

class MIL_AnswerTrigger(object):
  """ The Multiple Intance Learning (MIL) based Answer Triggering model.
  """
  def __init__(self, config, buckets,
         is_training=True, # If False, don't calculate gradients
         embedding_initializer=None):
    self.batch_size = batch_size = config.batch_size
    self.max_sentence_num = max_sentence_num = config.max_sentence_num
    self.max_question_length = max_question_length = config.max_question_length
    self.max_sentence_length = max_sentence_length = config.max_sentence_length
    self.global_step = tf.get_variable('gloabl_step', [],
        initializer=tf.constant_initializer(value=0, dtype=config.data_type),
        trainable=False)
    self._lr = tf.get_variable('learning_rate', [],
        initializer=tf.constant_initializer(value=config.learning_rate,
        dtype=config.data_type), trainable=False)
    self._lr_decay = config.lr_decay
    # self._lr_update = self._lr.assign(self._lr * self._lr_decay)
    self._lr_reset = self._lr.assign(float(config.learning_rate))
    self._nn_hidden_size = nn_hidden_size = config.nn_hidden_size
    self.eval_stat = EvalStats()

    # ================================
    # BUILD PLACEHOLDERS FOR QUESTIONS
    self._input_question = []     # NOTE: question tokens
    self._input_question_word_mask = [] # NOTE: padding mask on the question
    with tf.variable_scope("question"):
      self._input_question_feat = \
          tf.placeholder(tf.float32, shape=[batch_size, 1], name='qf') # NOTE: features of the question
      for i in xrange(max_question_length):
        self._input_question.append(
            tf.placeholder(tf.int32, shape=[batch_size], name="q-{0}".format(i)))
        self._input_question_word_mask.append(
            tf.placeholder(tf.float32, shape=[batch_size], name="qm-{0}".format(i)))

    # =============================================================
    # BUILD PLACEHOLDER FOR CANDIDATE ANSWER SENTENCES IN THE "BAG"
    self._input_sentences = []          # NOTE: each token in the sentences
    self._input_sentence_mask = []      # NOTE: padding mask on the whole sentences
    self._input_sentence_word_mask = [] # NOTE: padding mask on each word in the sentences
    self._input_sentence_feat = []      # NOTE: features of each the sentences
    with tf.variable_scope("sentences"):
      for i in xrange(max_sentence_num):
        self._input_sentence_mask.append(
            tf.placeholder(tf.float32, shape=[batch_size], name="sm-{0}".format(i)))
        self._input_sentence_feat.append(
            tf.placeholder(tf.float32, shape=[batch_size, 3], name="sf-{0}".format(i)))
        self._input_sentences.append([])
        self._input_sentence_word_mask.append([])
        for j in xrange(max_sentence_length):
          self._input_sentences[i].append(
              tf.placeholder(tf.int32, shape=[batch_size], name="s-{0}-{1}".format(i, j)))
          self._input_sentence_word_mask[i].append(
              tf.placeholder(tf.float32, shape=[batch_size], name="swm-{0}-{1}".format(i, j)))

    # ===============================================
    # BUILD PLACEHOLDER FOR TARGETS: BAG LEVEL LABELS
    self._targets = tf.placeholder(tf.float32, [batch_size], name="bag-label") # NOTE: bag label
    self._targets_mask = tf.placeholder(tf.float32, [batch_size], name='bag-label_mask') # NOTE: 1 if not all none-padding sentence have the same label
    self._sent_targets = []
    for i in xrange(max_sentence_num):
      self._sent_targets.append(
          tf.placeholder(tf.float32, shape = [batch_size],
          name="sent-label-{}".format(i)))

    # ==============
    # WORD EMBEDDING
    embed_question, embed_sentences = \
        self.embed_tokens(is_training, config, embedding_initializer)

    # ==================
    # QUESTION RNN MODEL
    self._question_vector_b, self._question_attention_b, self._question_output_b = \
        self.question_model_with_bucket(is_training, config, buckets, embed_question, self._input_question_word_mask)
    assert(len(self._question_vector_b) == len(buckets))    # list of [batch_size, hidden_size]
    assert(len(self._question_output_b) == len(buckets))    # list of [batch_size, q_len, hidden_size]
    assert(len(self._question_attention_b) == len(buckets)) # list of [batch_size, q_len] or [0] if DISABLE_ATTENTION

    # ===================
    # SENTENCES RNN MODEL
    self._sentence_vectors_b, self._sentence_attention_b, self._sentence_output_b = \
        self.sentence_model_with_bucket(is_training, config, buckets,
                embed_sentences,
                self._input_sentence_word_mask,
                # self._question_vector_b # NOTE: Cross attention
                )
    assert(len(self._sentence_vectors_b) == len(buckets))   # list of [batch_size, s_len, hidden_size]
    assert(len(self._sentence_output_b) == len(buckets))    # list of [batch_size, sent_num, s_len, hidden_size]
    assert(len(self._sentence_attention_b) == len(buckets)) # list of [batch_size, sent_num, s_len] or [0, sent_num] if DISABLE_ATTENTION

    # ===========================================
    # GROUP TOKEN-LEVEL PLACEHOLDERS INTO BUCKETS
    sentence_feat_b, sentence_mask_b, sentence_trgt_b = [], [], []
    for b, (sent_num, _, _) in enumerate(buckets):
      sentence_feat, sentence_mask, sentence_trgt = [], [], []
      for i in xrange(sent_num):
        sentence_feat.append(self._input_sentence_feat[i])
        sentence_mask.append(self._input_sentence_mask[i])
        sentence_trgt.append(self._sent_targets[i])
      sentence_feat_b.append(tf.pack(sentence_feat, 1)) # [batch_size, sent_num, 3]
      sentence_mask_b.append(tf.pack(sentence_mask, 1)) # [batch_size, sent_num]
      sentence_trgt_b.append(tf.pack(sentence_trgt, 1)) # [batch_size, sent_num]

    # =================================================================
    # LEARN THE FINAL PREDICTION OF ANSWER SELECTION AND TRIGGERING
    # USING ATTENTION BASED *OR* AVERAGE POOLING (IF DISABLE_ATTENTION)
    self._sent_pred_b, group_rslt = group_learning(
        is_training, config, buckets,
        self._question_vector_b, self._sentence_vectors_b,
        sentence_mask_b, sentence_trgt_b,
        self._targets, self._targets_mask,
        question_feat=self._input_question_feat,   # Question features: (length)
        sentence_feat_b=sentence_feat_b,     # Sentence features: (length and 2 word overlapping)
        )
    assert(len(buckets) == len(self._sent_pred_b)) # list of [batch_size, s_len]

    self._pred_b, self._cost_b, self._sent_cost_b, self._regu_b, \
        self._cost_list_b = group_rslt
    assert(len(buckets) == len(self._pred_b)) # list of [batch_size, sent_num]
    assert(len(buckets) == len(self._cost_b)) # list of [] (scalar)
    assert(len(buckets) == len(self._sent_cost_b)) # list of [batch_size, sent_num]
    assert(len(buckets) == len(self._regu_b)) # list of [] (scalar)

    # ===========================
    # DISPLAY TRAINABLE VARIABLES
    tvars = tf.trainable_variables()
    self._tvars = tvars
    print("\nAll trainable variables:")
    l2_regu_list = []
    for v in tvars:
      print(v.name + '\t' + str(v.get_shape()))
      if "Matrix" in v.name:
        # print("add l2 regularization")
        l2_regu_list.append(tf.nn.l2_loss(v))
    l2_regu = tf.reduce_sum(l2_regu_list)
    # l2_regu = tf.Print(l2_regu, [l2_regu_list, l2_regu], message='regularization', summarize=200)

    # ===============
    # OTHER VARIABLES
    other_vars = [self.global_step, self._lr]

    # self._saver = tf.train.Saver(tvars + other_vars)
    self._saver = tf.train.Saver(tf.all_variables())
    # self._saver = tf.train.Saver(tf.global_variables())
    # pdb.set_trace()

    self._q_fw_rnn_saver = tf.train.Saver(tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES,
        scope="model/question_rnn/q_vector/attention_RNN/BiRNN/FW"))
    self._q_bw_rnn_saver = tf.train.Saver(tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES,
        scope="model/question_rnn/q_vector/attention_RNN/BiRNN/BW"))
    self._s_fw_rnn_saver = tf.train.Saver(tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES,
        scope="model/sentence_rnn/s_vectors/attention_RNN/BiRNN/FW"))
    self._s_bw_rnn_saver = tf.train.Saver(tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES,
        scope="model/sentence_rnn/s_vectors/attention_RNN/BiRNN/BW"))

    if not is_training: return

    # Training operators
    self._train_op_b = []
    optimizer = tf.train.AdadeltaOptimizer(self._lr, self._lr_decay)

    for b in xrange(len(buckets)):
      if NUMERIC_CHECK:
        self._cost_b[b] = tf.check_numerics(self._cost_b[b], "cost")
      if NUMERIC_CHECK:
        self._difficulty_cost_b[b] = \
            tf.check_numerics(self._difficulty_cost_b[b], "cost")

      final_cost = (self._cost_b[b] + # Cost based on bags of prediction
                    config.l2_regu_weight * l2_regu)

      self._train_op_b.append(optimizer.minimize(final_cost,
                                                 global_step=self.global_step
                                                 ))

  def question_model_with_bucket(self, is_training, config, buckets,
                                 embed_question, question_mask):
    """
    """
    question_vector_b = []
    question_attention_b = []
    question_outputs_b = []

    with tf.variable_scope("question_rnn"):
      for b_idx, (sent_num, ques_len, sent_len) in enumerate(buckets):
        bucket_embed_question = []
        bucket_mask_question = []
        for i in xrange(ques_len):
          bucket_embed_question.append(embed_question[i])
          bucket_mask_question.append(question_mask[i])

        with tf.variable_scope("q_vector", reuse=True if b_idx > 0 else None):
          vector, attention, outputs = qa_rnn(config, is_training,
                                              [bucket_embed_question],
                                              [bucket_mask_question])
          assert(vector.get_shape() == (self.batch_size, 1, 2*config.rnn_hidden_size))
          question_vector_b.append(vector[:,0,:])
          assert(outputs.get_shape() == (self.batch_size, 1, ques_len, 2*config.rnn_hidden_size))
          question_outputs_b.append(outputs[:,0,:,:])
          if POOLING == 'attention':
            assert(attention.get_shape() == (self.batch_size, 1, ques_len))
          question_attention_b.append(attention[:,0])

    return question_vector_b, question_attention_b, question_outputs_b


  def sentence_model_with_bucket(self, is_training, config, buckets,
                                 embed_sentences, sentences_mask,
                                 question_vector_b=None):
    """ Args:
    """
    sentence_vectors_b = []
    sentence_attention_b = []
    sentence_outputs_b = []

    with tf.variable_scope("sentence_rnn"):
      for b_idx, (sent_num, ques_len, sent_len) in enumerate(buckets):
        bucket_embed_sentences = []
        bucket_mask_sentences = []
        for i in xrange(sent_num):
          bucket_embed_sentences.append([])
          bucket_mask_sentences.append([])
          for j in xrange(sent_len):
            bucket_embed_sentences[i].append(embed_sentences[i][j])
            bucket_mask_sentences[i].append(sentences_mask[i][j])

        with tf.variable_scope("s_vectors", reuse=True if b_idx > 0 else None):
          if question_vector_b is not None: xvector = question_vector_b[b_idx]
          else: xvector = None


          vectors, attention, outputs = qa_rnn(config, is_training,
                                               bucket_embed_sentences,
                                               bucket_mask_sentences,
                                               xvector=xvector)

          assert(vectors.get_shape() == (self.batch_size, sent_num, 2*config.rnn_hidden_size))
          sentence_vectors_b.append(vectors)
          assert(outputs.get_shape() == (self.batch_size, sent_num, sent_len, 2*config.rnn_hidden_size))
          sentence_outputs_b.append(outputs)
          if POOLING == 'attention':
            assert(attention.get_shape() == (self.batch_size, sent_num, sent_len))
          sentence_attention_b.append(attention)

    return sentence_vectors_b, sentence_attention_b, sentence_outputs_b

  def embed_tokens(self, is_training, config, embedding_initializer):
    """Embedds input tokens.
    """
    vocab_size = config.vocab_size
    size = config.word_embed_size
    max_question_length = self.max_question_length
    max_sentence_length = self.max_sentence_length
    max_sentence_num = self.max_sentence_num

    with tf.variable_scope("embed"):
      with tf.device("/cpu:0"):
        embedding = tf.get_variable(
            "embedding_mat", [vocab_size, size],
            initializer=embedding_initializer,
            dtype=config.data_type,
            trainable=False # Continue to train pretrained word2vec
            # trainable=True # Continue to train pretrained word2vec
            )

        self._embedding = embedding
        embed_question= []
        for i in xrange(max_question_length):
          embed_question.append(
              tf.nn.embedding_lookup(embedding, self._input_question[i]))
          if is_training and config.w_embed_keep_prob < 1:
            embed_question[i] = tf.nn.dropout(embed_question[i],
                                              config.w_embed_keep_prob)
          if NUMERIC_CHECK:
            embed_question[i] = \
                tf.check_numerics(embed_question[i],
                    "embed_question[{}][{}] numeric error".format(i))

        embed_sentences = []
        for i in xrange(max_sentence_num):
          embed_sentences.append([])
          for j in xrange(max_sentence_length):
            embed_sentences[i].append(
              tf.nn.embedding_lookup(embedding, self._input_sentences[i][j]))
            if is_training and config.w_embed_keep_prob < 1:
              embed_sentences[i][j] = tf.nn.dropout(embed_sentences[i][j],
                                                    config.w_embed_keep_prob)
            if NUMERIC_CHECK:
              embed_sentences[i][j] = \
                  tf.check_numerics(embed_sentences[i][j],
                      "embed_sentences[{}][{}] numeric error".format(i, j))

    return embed_question, embed_sentences


  # RESULTS
  @property
  def sent_pred_b(self):
    return self._sent_pred_b

  @property
  def pred_b(self):
    return self._pred_b

  @property
  def cost_b(self):
    return self._cost_b

  @property
  def cost_list_b(self):
    return self._cost_list_b

  @property
  def sent_cost_b(self):
    return self._sent_cost_b

  @property
  def regu_b(self):
    return self._regu_b

  # INPUTS
  @property
  def input_question(self):
    return self._input_question

  @property
  def input_question_feat(self):
    return self._input_question_feat

  @property
  def input_sentences(self):
    return self._input_sentences

  @property
  def input_question_word_mask(self):
    return self._input_question_word_mask

  @property
  def input_sentence_word_mask(self):
    return self._input_sentence_word_mask

  @property
  def input_sentence_feat(self):
    return self._input_sentence_feat
  
  @property
  def input_sentence_mask(self):
    return self._input_sentence_mask

  @property
  def targets(self):
    return self._targets

  @property
  def targets_mask(self):
    return self._targets_mask
    
  @property
  def sent_targets(self):
    return self._sent_targets

  # OTHERS
  @property
  def lr(self):
    return self._lr

  @property
  def lr_reset(self):
    return self._lr_reset

  @property
  def train_op_b(self):
    return self._train_op_b

  @property
  def print_op(self):
    return self._print_op

  @property
  def tvars(self):
    return self._tvars

  @property
  def q_fw_rnn_saver(self):
    return self._q_fw_rnn_saver

  @property
  def q_bw_rnn_saver(self):
    return self._q_bw_rnn_saver

  @property
  def s_fw_rnn_saver(self):
    return self._s_fw_rnn_saver

  @property
  def s_bw_rnn_saver(self):
    return self._s_bw_rnn_saver

  @property
  def saver(self):
    return self._saver

  @property
  def question_attention_b(self):
    return self._question_attention_b

  @property
  def sentence_attention_b(self):
    return self._sentence_attention_b

  @property
  def embedding(self):
    return self._embedding
