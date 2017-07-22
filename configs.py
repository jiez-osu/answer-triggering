# Author: Jie Zhao
#===============================================================================
NUMERIC_CHECK = False
NUMERIC_CHECK_ALL = False

class Config(object):
  """Small config."""
  # Model ralated configurations
  init_scale = 0.0 # basically controls the initializer of RNN
  learning_rate = 0.1
  max_grad_norm = 5
  num_layers = 1
  word_embed_size = 300
  rnn_hidden_size = 200
  nn_hidden_size = 400
  w_embed_keep_prob = 0.5
  rnn_keep_prob = 0.5
  keep_prob = 0.5
  lr_decay = 0.95
  batch_size = 20
  vocab_size = 0    # to be filled in
  data_type = None  # to be filled in
  # Data related configurations
  max_epoch = 5
  max_max_epoch = 100
  max_sentence_num = 0  # to be filled in when using buckets
  max_question_length = 0  # to be filled in when using buckets
  max_sentence_length = 0  # to be filled in when using buckets
  use_checkpoint = False
  # Cost function hyper paramters
  alpha = 1.0
  beta  = 1.2
  gamma = 1.0
  # Margins used in hinge losses
  pos_margin = 0.2 # positive prediction margin to 0.5
  neg_margin = 0.3 # negative prediction margin to 0.5
  pos_neg_margin = 0.5 # postitive and negative margin gap
  # Regularization term hyper parameters
  l2_regu_weight = 1e-4
  max_gap = 0.3  # max prediction gap for same label instances in a bag
  min_gap = 0.7  # min prediction gap for diff label instances in a bag
  phi = 1.0
  # Combine features
  plus_cnt = False
  plus_qlen = False
  plus_slen = False


class SelfTestConfig(object):
  """Tiny config, for testing."""
  # Model related configurations
  init_scale = 0.1
  learning_rate = 0.5
  max_grad_norm = 0.5
  num_layers = 1
  word_embed_size = 16
  rnn_hidden_size = 32
  nn_hidden_size = 12
  w_embed_keep_prob = 0.5
  rnn_keep_prob = 0.5
  keep_prob = 0.5
  lr_decay = 0.5
  batch_size = 2
  vocab_size = 10
  data_type = None
  # Data related configurations
  max_epoch = 5
  max_max_epoch = 30
  max_sentence_num = 0  # to be filled in when using buckets
  max_question_length = 0  # to be filled in when using buckets
  max_sentence_length = 0  # to be filled in when using buckets
  use_checkpoint = True
  # Cost function hyper paramters
  alpha = 0.5
  beta  = 1.0
  gamma = 0.5
  # Margins used in hinge losses
  pos_margin = 0.1 # positive prediction margin to 0.5
  neg_margin = 0.1 # negative prediction margin to 0.5
  pos_neg_margin = 0.1 # postitive and negative margin gap
  # Regularization term hyper parameters
  l2_regu_weight = 1e-4
  max_gap = 0.4  # max prediction gap for same label instances in a bag
  min_gap = 0.5  # min prediction gap for diff label instances in a bag
  phi = 1.0
  # Combine features
  plus_cnt = False
  plus_qlen = False
  plus_slen = False

def print_config(config):
  """Print config to output.
  Args:
    config: a SmallConfig, MediumConfig, LargeConfig, TestConfig or
      SelfTestConfig object.
  """
  print("\n----- ----- configurations ----- -----")
  # Model related configurations
  print("{:20s}: {}".format("init_scale", config.init_scale))
  print("{:20s}: {}".format("learning_rate", config.learning_rate))
  print("{:20s}: {}".format("max_grad_norm", config.max_grad_norm))
  print("{:20s}: {}".format("num_layers", config.num_layers))
  print("{:20s}: {}".format("word_embed_size", config.word_embed_size))
  print("{:20s}: {}".format("rnn_hidden_size", config.rnn_hidden_size))
  print("{:20s}: {}".format("nn_hidden_size", config.nn_hidden_size))
  print("{:20s}: {}".format("w_embed_keep_prob", config.w_embed_keep_prob))
  print("{:20s}: {}".format("rnn_keep_prob", config.rnn_keep_prob))
  print("{:20s}: {}".format("keep_prob", config.keep_prob))
  print("{:20s}: {}".format("lr_decay", config.lr_decay))
  print("{:20s}: {}".format("batch_size", config.batch_size))
  print("{:20s}: {}".format("vocab_size", config.vocab_size))
  print("{:20s}: {}".format("data_type", config.data_type))
  # Data related configurations
  print("{:20s}: {}".format("max_epoch", config.max_epoch))
  print("{:20s}: {}".format("max_max_epoch", config.max_max_epoch))
  print("{:20s}: {}".format("max_sentence_num", config.max_sentence_num))
  print("{:20s}: {}".format("max_question_length", config.max_question_length))
  print("{:20s}: {}".format("max_sentence_length", config.max_sentence_length))
  print("{:20s}: {}".format("use_checkpoint", config.use_checkpoint))
  # Cost term weights
  print("{:20s}: {}".format("alpha", config.alpha))
  print("{:20s}: {}".format("beta", config.beta))
  print("{:20s}: {}".format("gamma", config.gamma))
  # Margins
  print("{:20s}: {}".format("pos_margin", config.pos_margin))
  print("{:20s}: {}".format("neg_margin", config.neg_margin))
  print("{:20s}: {}".format("pos_neg_margin", config.pos_neg_margin))
  # Regularization parameters
  print("{:20s}: {}".format("l2_regu_weight", config.l2_regu_weight))
  print("{:20s}: {}".format("max_gap", config.max_gap))
  print("{:20s}: {}".format("min_gap", config.min_gap))
  print("{:20s}: {}".format("phi", config.phi))
  # Plus features
  print("{:20s}: {}".format("plus_cnt", config.plus_cnt))
  print("{:20s}: {}".format("plus_qlen", config.plus_qlen))
  print("{:20s}: {}".format("plus_slen", config.plus_slen))

  print("")
  # raw_input("Press enter to continue.")
