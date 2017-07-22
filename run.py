# Author: Jie Zhao
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

from configs import Config, SelfTestConfig, print_config,  NUMERIC_CHECK, \
                    NUMERIC_CHECK_ALL
from model import MIL_AnswerTrigger
from my_data import MyData 
import random

from run_model import train_model, test_model, save_checkpoint
import pdb
from toy_data import fake_data_with_buckets
from prep_wikiqa_data import prep_data, BUCKETS, BUCKETS_COLLAPSED, WordVecs

flags = tf.flags
logging = tf.logging

flags.DEFINE_bool("use_fp16", False,
    "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_string("summaries_dir", "./train",
    "Direcory that stores tensor board data.")
flags.DEFINE_string("ckpt_dir", "./checkpoints/checkpoints",
    "Directort that stores model variable checkpoints")
flags.DEFINE_integer("epochs_per_validation", 1,
    "Training epoches between two validation")
flags.DEFINE_integer("epochs_per_checkpoint", 100,
    "Training epoches between two checkpoint savings")
flags.DEFINE_integer("steps_per_display", 20,
    "Training steps between two display of answer prediction results using the validation set")
flags.DEFINE_bool("self_test", False, "Run a self-test if this is set to true.")
flags.DEFINE_bool("train", False, "Train the model and run validation.")
flags.DEFINE_bool("test", False,"Test.")
flags.DEFINE_bool("continue_training", False,
    "If continue training from a previous checkpoint.")
# MODEL HYPER PARAMETERS THAT NEEDS SELECTION
flags.DEFINE_float("learning_rate", -1.0, "overwrite learning rate if non-negative")
flags.DEFINE_float("alpha", -1.0, "overwrite alpha if non-negative")
flags.DEFINE_float("beta", -1.0, "overwrite beta if non-negative")
flags.DEFINE_float("pos_margin", -1.0, "overwrite pos_margin if non-negative")
flags.DEFINE_float("neg_margin", -1.0, "overwrite neg_margin if non-negative")
flags.DEFINE_float("pos_neg_margin", -1.0, "overwrite pos_neg_margin if non-negative")
flags.DEFINE_string("ckpt_suffix", "",
		"suffix that are added to checkpoint")
flags.DEFINE_bool("plus_cnt", False, "overwrite plus_cnt")
flags.DEFINE_bool("plus_qlen", False, "overwrite plus_qlen")
flags.DEFINE_bool("plus_slen", False, "overwrite plus_slen")
flags.DEFINE_integer("random_seed", 11, "Random seed.")
flags.DEFINE_integer("tf_random_seed", 11, "Random seed.")
FLAGS = flags.FLAGS

def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32

def overwrite_config(config):
  if FLAGS.learning_rate > 0: config.learning_rate = FLAGS.learning_rate
  if FLAGS.alpha > 0: config.alpha = FLAGS.alpha
  if FLAGS.beta > 0: config.beta = FLAGS.beta
  if FLAGS.pos_margin > 0: config.pos_margin = FLAGS.pos_margin
  if FLAGS.neg_margin > 0: config.neg_margin = FLAGS.neg_margin
  if FLAGS.pos_neg_margin > 0: config.pos_neg_margin = FLAGS.pos_neg_margin
  if FLAGS.plus_cnt: config.plus_cnt = True
  if FLAGS.plus_qlen: config.plus_qlen = True
  if FLAGS.plus_slen: config.plus_slen = True

def get_config():
  if FLAGS.self_test == True: config = SelfTestConfig()
  else: config = Config()
  overwrite_config(config)
  return config

def test():
  """Test the model."""
  buckets = BUCKETS
  train_tuple_b, valid_tuple_b, test_tuple_b, word2id, word_embeddings = \
      prep_data()
  mydata = MyData(train_tuple_b, valid_tuple_b, test_tuple_b, word2id,
                  word_embeddings)
  assert(mydata.test_buckets == buckets)
  buckets_t = zip(*buckets)

  # Create model
  config = get_config()
  config.vocab_size = len(word2id)
  config.max_sentence_num = max(buckets_t[0])
  config.max_question_length = max(buckets_t[1])
  config.max_sentence_length = max(buckets_t[2])
  config.data_type = data_type()
  config.batch_size = 1
  config.init_scale = np.sqrt(6.0 / (config.word_embed_size + config.rnn_hidden_size))
  print_config(config)

  with tf.Session() as session:
    # Set random seed to fixed number
    tf.set_random_seed(FLAGS.tf_random_seed)
    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)

    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    embedding_initializer = tf.constant_initializer(word_embeddings,
                                                    dtype=config.data_type)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
      m = MIL_AnswerTrigger(config=config, buckets=buckets,
                            is_training=False,
                            embedding_initializer=embedding_initializer)
    test_model(FLAGS, session, m, config, mydata, if_test=True,
               if_show_qual_rslt=True, if_load_ckpt=True)


def train():
  """Train the model."""
  buckets = BUCKETS
  train_tuple_b, valid_tuple_b, test_tuple_b, word2id, word_embeddings = \
      prep_data()
  mydata = MyData(train_tuple_b, valid_tuple_b, test_tuple_b,
                  word2id, word_embeddings)
  assert(mydata.train_buckets == buckets)
  assert(mydata.valid_buckets == buckets)
  buckets_t = zip(*buckets)

  # Create model
  config = get_config()
  config.vocab_size = len(word2id)
  config.max_sentence_num = max(buckets_t[0])
  config.max_question_length = max(buckets_t[1])
  config.max_sentence_length = max(buckets_t[2])
  config.data_type = data_type()
  config.init_scale = np.sqrt(6.0 / (config.word_embed_size + config.rnn_hidden_size))
  print_config(config)
  eval_config = get_config()
  eval_config.vocab_size = len(word2id)
  eval_config.max_sentence_num = max(buckets_t[0])
  eval_config.max_question_length = max(buckets_t[1])
  eval_config.max_sentence_length = max(buckets_t[2])
  eval_config.data_type = data_type()
  eval_config.batch_size = 10
  eval_config.init_scale = np.sqrt(6.0 / (config.word_embed_size + config.rnn_hidden_size))

  with tf.Session() as session:
    # Set random seed to fixed number
    tf.set_random_seed(FLAGS.tf_random_seed)
    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)

    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    embedding_initializer = tf.constant_initializer(word_embeddings,
                                                    dtype=config.data_type)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
      m = MIL_AnswerTrigger(config=config, buckets=buckets,
                            embedding_initializer=embedding_initializer)
    with tf.variable_scope("model", reuse=True, initializer=initializer):
      m_valid = MIL_AnswerTrigger(config=eval_config, buckets=buckets,
                                  is_training=False,
                                  embedding_initializer=embedding_initializer)

    train_model(FLAGS, session, m, config, mydata, m_valid)


def self_test_train():
  """ Self test the model with toy data, training part.
  """
  # Load and prepare fake data
  buckets, buckets_collapsed, bucket_sizes, train_tuple_b, word2id, word_embeddings = \
      fake_data_with_buckets()
  mydata = MyData(train_tuple_b, train_tuple_b, train_tuple_b, word2id, word_embeddings)
  assert(mydata.train_buckets == buckets)
  assert(mydata.train_bucket_sizes == bucket_sizes)
  assert(mydata.valid_buckets == buckets)
  assert(mydata.valid_bucket_sizes == bucket_sizes)
  buckets_t = zip(*buckets)

  # Create model with vocabulary of 10, 2 small buckets
  config = SelfTestConfig()
  config.vocab_size = 10
  config.max_sentence_num = max(buckets_t[0])
  config.max_question_length = max(buckets_t[1])
  config.max_sentence_length = max(buckets_t[2])
  config.data_type = data_type()
  config.init_scale = np.sqrt(6.0 / (config.word_embed_size + config.rnn_hidden_size))
  eval_config = SelfTestConfig()
  eval_config.vocab_size = 10
  eval_config.max_question_length = max(buckets_t[1])
  eval_config.max_sentence_num = max(buckets_t[0])
  eval_config.max_sentence_length = max(buckets_t[2])
  eval_config.data_type = data_type()
  eval_config.batch_size = 2
  eval_config.init_scale = np.sqrt(6.0 / (config.word_embed_size + config.rnn_hidden_size))

  with tf.Session() as session:
    # Set random seed to fixed number
    tf.set_random_seed(FLAGS.tf_random_seed)
    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)

    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    embedding_initializer = tf.constant_initializer(word_embeddings,
                                                    dtype=config.data_type)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
      m = MIL_AnswerTrigger(config=config, buckets=buckets,
                            embedding_initializer=embedding_initializer)
    with tf.variable_scope("model", reuse=True, initializer=initializer):
      m_valid = MIL_AnswerTrigger(config=eval_config, buckets=buckets,
                                  is_training=False,
                                  embedding_initializer=embedding_initializer)

    train_model(FLAGS, session, m, config, mydata, m_valid)
    save_checkpoint(FLAGS, session, m)


def self_test_test():
  """ Self test the model with toy data, testing part.
  """
  # Load and prepare fake data
  buckets, bucket_collapsed, bucket_sizes, train_tuple_b, word2id, \
      word_embeddings = fake_data_with_buckets()
  mydata = MyData(train_tuple_b, train_tuple_b, train_tuple_b,
                  word2id, word_embeddings)
  assert(mydata.train_buckets == buckets)
  assert(mydata.train_bucket_sizes == bucket_sizes)
  assert(mydata.valid_buckets == buckets)
  assert(mydata.valid_bucket_sizes == bucket_sizes)
  buckets_t = zip(*buckets)

  # Create model with vocabulary of 10, 2 small buckets
  config = SelfTestConfig()
  config.vocab_size = 10
  config.max_question_length = max(buckets_t[1])
  config.max_sentence_num = max(buckets_t[0])
  config.max_sentence_length = max(buckets_t[2])
  config.data_type = data_type()
  config.batch_size = 2
  config.init_scale = np.sqrt(6.0 / (config.word_embed_size + config.rnn_hidden_size))
  with tf.Session() as session:
    # Set random seed to fixed number
    tf.set_random_seed(FLAGS.tf_random_seed)
    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)

    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    embedding_initializer = tf.constant_initializer(word_embeddings,
                                                    dtype=config.data_type)
    with tf.variable_scope("model", reuse=False, initializer=initializer):
      m_test = MIL_AnswerTrigger(config=config, buckets=buckets,
                                 is_training=False,
                                 embedding_initializer=embedding_initializer)
    test_model(FLAGS, session, m_test, config, mydata, if_test=True,
               if_show_qual_rslt=True, if_load_ckpt=True)


def main(_):
  if FLAGS.self_test:
    if FLAGS.train:
      self_test_train()
    elif FLAGS.test:
      self_test_test()
    else:
      print("Error: must set a mode: train | test")
  else:
    if FLAGS.train:
      train()
    elif FLAGS.test:
      test()
    else:
      print("Error: must set a mode: train | test")


if __name__ == "__main__":
  tf.app.run()
