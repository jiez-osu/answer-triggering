# Copyright Jie Zhao
#===============================================================================
import tensorflow as tf
import pdb
from configs import NUMERIC_CHECK
import numpy as np

def group_learning(is_training, config, buckets,
    question_vector_b,     # list of [batch_size, hidden_size]
    sentence_vectors_b,  # list of [batch_size, sent_num, hidden_size]
    sentence_mask_b,       # list of [batch_size, sent_num]
    sentence_trgt_b,       # list of [batch_size, sent_num]
    trgt,                # [batch_size]
    trgt_mask,             # [batch_size]
    question_feat=None,  # [batch_size, 1]
    sentence_feat_b=None, # [batch_size, sent_num, 3]
    q_difficulty_b=None,  # list of [batch_size]
    sim_feat_b=None):    # list of [abtch_size, sent_num, 3]
  """ Multi Instance Learing, Take LSTM output as inputs.
  """
  sent_preds_b = []
  AS_cost_b = []
  AS_sent_cost_b = []
  pred_b = []
  cost_b = []
  sent_cost_b = []
  regu_b = []
  cost_list_b = []
  reuse_flag = False
  for b, (sent_num, q_len, s_len) in enumerate(buckets):
    if b > 0: reuse_flag = True
    q_embedding = question_vector_b[b]
    s_embedding = sentence_vectors_b[b]
    q_feats = question_feat
    s_feats = (None if sentence_feat_b is None else sentence_feat_b[b])
    q_difficulty = (None if q_difficulty_b is None else q_difficulty_b[b])
    sim_feat = (None if sim_feat_b is None else sim_feat_b[b])

    # PASS OUTPUTS THROUGH A FULLY CONNECTED PERCEPTRON LAYER AND THEN LOGISTIC REGRESSION
    sent_logits, sent_preds = mix_and_pred(is_training, config, reuse_flag, sent_num,
                   q_embedding,  # [batch_dize, hidden_size]
                   s_embedding,  # [batch_size, sent_num, hidden_size]
                   q_feats=q_feats, # [batch_size, 1]
                   s_feats=s_feats, # [batch_size, sent_num, 3]
                   q_difficulty=q_difficulty, # [batch_size]
                   # sim_feat=sim_feat,       # [batch_size, sent_num, 3]
                   )

    sent_preds_b.append(sent_preds) # [batch_size, sent_num]

    pred, cost, sent_cost, regu, cost_list = group_pick(config, sent_num, sent_logits, sent_preds, sentence_mask_b[b], sentence_trgt_b[b], trgt, trgt_mask)
    # pred, cost, sent_cost, regu, cost_list = group_pick_naive(config, sent_num, sent_logits, sent_preds, sentence_mask_b[b], sentence_trgt_b[b], trgt, trgt_mask)

    if sim_feat is not None:
      sim_regu = group_sim_regu(config, reuse_flag, sent_num,
            sentence_mask_b[b], sentence_trgt_b[b],
            trgt, trgt_mask, sim_feat)
    else:
      sim_regu = tf.constant(0, dtype=config.data_type)

    pred_b.append(pred)
    cost_b.append(cost)
    sent_cost_b.append(sent_cost)
    regu_b.append(regu + sim_regu)
    cost_list_b.append(cost_list)

  return sent_preds_b, (pred_b, cost_b, sent_cost_b, regu_b, cost_list_b)


#===============================================================================
# METHODS THAT INCORPORATE QUESTION AND ANSWER VECTOR REPRESENTAION INTO
# ANSWER SELECTION SCORES
#===============================================================================

def mix_and_pred(is_training, config, if_reuse, sent_num,
    q_vector,         # [batch_size, hidden_size]
    s_vectors,          # [batch_size, sent_num, hidden_size]
    q_feats = None,   # None or [batch_size, 1]
    s_feats = None,   # None or [batch_size, sent_num, 3]
    q_difficulty=None, # None or [batch_size]
    sim_feat=None):   # None of [batch_size, sent_num, 3]
  """ Mix up question and sentence RNN outputs to predict if it's a true
      QA pair. Use a fully connected feed forward NN architecture.
  """
  # CONCATENATE QUESTION AND SENTENCE RNN OUTPUT VECTORS
  feat_vectors = []
  for i in xrange(sent_num):
    feat_list = [q_vector, s_vectors[:,i,:]]
    feat_dims = config.rnn_hidden_size * 2 * 2
    if q_difficulty is not None:
      feat_list.append(tf.expand_dims(q_difficulty, axis=1)) # [batch_size, 1]
      feat_dims += 1
    # if sim_feat is not None:
    #   feat_list.append(sim_feat[:,i]) # [batch_size, 3]
    #   feat_dims += 3
    feat_vectors.append(tf.concat(1, feat_list))
  assert(sent_num == len(feat_vectors))

  # A FULLY CONNECTED LAYER(S)
  with tf.variable_scope("hidden_layer", reuse=True if if_reuse else None):
    hidden_output = perceptron(is_training, config,
          input_vecs=feat_vectors, # list of [batch_size, input_size]
          input_size=feat_dims,
          hidden_size=config.nn_hidden_size,
          scope="hidden_1") # list of [batch_size, hidden_size]
    assert(sent_num == len(hidden_output))
    
    hidden_output_1 = perceptron(is_training, config,
          input_vecs=hidden_output, # list of [batch_size, hidden_size]
          input_size=config.nn_hidden_size,
          hidden_size=config.nn_hidden_size,
          scope='hidden_2') # list of [batch_size, hidden_size]
    assert(sent_num == len(hidden_output_1))

  # A LOGISTIC REGRESSION LAYER
  with tf.variable_scope("logis_regres", reuse=True if if_reuse else None):
    input_vecs = [[v] for v in hidden_output_1]
    input_size = config.nn_hidden_size
    input_size_plus = 0
    # if q_feats is not None:
    if config.plus_qlen:
      input_size_plus += 1
      for i in xrange(sent_num):
        input_vecs[i].append(q_feats) # list of [batch_size, input_size] 
    # if s_feats is not None:
    if config.plus_slen:
      input_size_plus += 1
      for i in xrange(sent_num):
        input_vecs[i].append(tf.expand_dims(s_feats[:,i,0], axis=1)) # list of [batch_size, input_size]
    if config.plus_cnt:
      input_size_plus += 2
      for i in xrange(sent_num):
        input_vecs[i].append(s_feats[:,i,1:3])

    input_vecs = [tf.concat(1, v) for v in input_vecs]        
    logits, preds = logistic_regression(config, input_vecs=input_vecs, 
                        input_size=input_size+input_size_plus, zero_suffix_size=input_size_plus)
    assert(sent_num == len(logits))
    assert(sent_num == len(preds))

  logits = tf.pack(logits, axis=1) # [batch_size, sent_num]
  preds = tf.pack(preds, axis=1)   # [batch_size, sent_num]
  return logits, preds


def perceptron(is_training, config,
         input_vecs, # list of [batch_size, input_size]
         input_size,
         hidden_size,
         act_func=tf.tanh, # default activation function
         scope='hidden'):
  """ Pass through a fully connected hidden layer
  """
  with tf.variable_scope(scope):
    outputs = []
    init_scale = np.sqrt(6.0 / (input_size + hidden_size))
    hidden_weight = tf.get_variable('Matrix',
        shape=[input_size, hidden_size], dtype=config.data_type,
        initializer=tf.random_uniform_initializer(
          -init_scale,init_scale, dtype=config.data_type))
    hidden_bias = tf.get_variable('Biases',
        shape=[hidden_size], dtype=config.data_type,
        initializer=tf.constant_initializer(value=0, dtype=config.data_type))

    for i in xrange(len(input_vecs)):
      output_vec = act_func(tf.matmul(input_vecs[i], hidden_weight) + hidden_bias)

      if is_training and config.keep_prob < 1: # Follows Rao and He et al., 2016
        output_vec = tf.nn.dropout(output_vec, keep_prob=config.keep_prob)
      outputs.append(output_vec)
  return outputs # list of [batch_size, hidden_size]


def logistic_regression(config,
      input_vecs, # list of [batch_size, input_size]
      input_size,
                        zero_suffix_size=0): # Initialize this number of values at the end of "Matrix" to zero
  """ The logistic regression layer for binary prediction.
  """
  sent_logits, sent_preds = [], [] # sentence level logit and prediction
  init_scale = np.sqrt(6.0 / (input_size + 1))
  assert(zero_suffix_size >= 0 and zero_suffix_size < input_size)
  # if zero_suffix_size == 0:
  if False:
    softmax_w = tf.get_variable("Matrix", [input_size, 1],
                                dtype=config.data_type,
                                initializer=tf.random_normal_initializer(
                                    -init_scale, init_scale, dtype=config.data_type))
  else:
    softmax_w_1 = tf.get_variable("Matrix_1", [input_size - zero_suffix_size, 1],
              dtype=config.data_type,
            initializer=tf.random_normal_initializer(
              -init_scale, init_scale, dtype=config.data_type))
    softmax_w_2 = tf.get_variable("Matrix_2", [zero_suffix_size, 1],
              dtype=config.data_type,
            initializer=tf.random_normal_initializer(
              -init_scale*1e-2, init_scale*1e-2, dtype=config.data_type))
            #initializer=tf.constant_initializer(value=0, dtype=config.data_type))
    softmax_w = tf.concat(0, [softmax_w_1, softmax_w_2])

  softmax_b = tf.get_variable("Bias", [1], dtype=config.data_type,
        initializer=tf.constant_initializer(value=0,
        dtype=config.data_type))

  for i in xrange(len(input_vecs)):
    logit = tf.squeeze(tf.matmul(input_vecs[i], softmax_w) + softmax_b, squeeze_dims=[1]) # [batch_size]
    pred = tf.sigmoid(logit, name="pred") # [batch_size]
    sent_logits.append(logit)
    sent_preds.append(pred)

  return sent_logits, sent_preds


def group_pick_naive(config, sent_num,
             logits, # [batch_size, sent_num]
             preds,  # [batch_size, sent_num]
             sentence_mask, # [batch_size, sent_num]
             sentence_trgt, # [batch_size, sent_num]
             trgt, # [batch_size]
             trgt_mask, # [batch_size]
             ):
  """ Calculate predication and cost for the *bag*
  Choose the closest one to represent the bag for training now.
  """
  sent_costs = []
  cost_list = [] # Different parts/terms in the overall cost function

  mask_asserts = []

  pos_mask = sentence_trgt * sentence_mask   # [batch_size, sent_num] # positive sentences
  neg_mask = (1.0 - sentence_trgt) * sentence_mask # [batch_size, sent_num] # negative sentences
  mask_asserts.append(tf.assert_non_negative(pos_mask))
  pos_bag_mask = trgt       # [batch_size] # positive bag
  neg_bag_mask = 1.0 - trgt # [batch_size] # negative bag

  # ============================
  # COST #1: FOR NEGATIVE BAGS
  # MAKE THE MAXIMUM SCORE < 0.5
  cost1_asserts = []
  preds_pad2zero = preds * sentence_mask # [batch_size, sent_num]
  max_pred = tf.reduce_max(preds_pad2zero, 1, name='max_pred') # [batch_size]
  cost1 = neg_bag_mask * tf.reduce_max(tf.pack(
      [tf.constant(0, shape=[config.batch_size], dtype=config.data_type),
       config.neg_margin - (0.5 - max_pred)], axis=1), 1) # [batch_size]
  #cost1_asserts.append(tf.assert_non_negative(cost1))
  with tf.control_dependencies(cost1_asserts + mask_asserts):
    avg_cost1 = tf.reduce_sum(cost1) / (tf.reduce_sum(neg_bag_mask) + 1e-12)

  # ============================================
  # COST #2: FOR POSITIVE BAGS
  # MAKE SURE THE LARGEST IS A POSITIVE INSTANCE
  # cost2_asserts = []
  # max_pos_pred = tf.reduce_max(preds * pos_mask, 1) # [batch_size]
  # max_neg_pred = tf.reduce_max(preds * neg_mask, 1) # [batch_size]
  # cost2 = pos_bag_mask * tf.reduce_max(tf.pack(
  #     [tf.constant(0, shape=[config.batch_size], dtype=config.data_type),
  #      config.pos_neg_margin - (max_pos_pred - max_neg_pred)], axis=1), 1)
  # cost2_asserts.append(tf.assert_non_negative(cost2))
  # cost2_asserts.append(tf.assert_equal(max_neg_pred * neg_bag_mask, max_pred * neg_bag_mask))
  # with tf.control_dependencies(cost2_asserts):
  #    avg_cost2 = tf.reduce_sum(cost2) / (tf.reduce_sum(pos_bag_mask) + 1e-12)

  # ===============================================================
  # COST #3: ALSO FOR POSITIVE BAGS
  # GET THE LARGEST FROM A POSITIVE SENTS; MAKE SURE IT'T ABOVE 0.5
  cost3_asserts = []
  cost3 = pos_bag_mask * tf.reduce_max(tf.pack(
      [tf.constant(0, shape=[config.batch_size], dtype=config.data_type),
       config.pos_margin - (max_pred - 0.5)], axis=1), 1) # [batch_size]
  #cost3_asserts.append(tf.assert_non_negative(cost3))
  with tf.control_dependencies(cost3_asserts):
    avg_cost3 = tf.reduce_sum(cost3) / (tf.reduce_sum(pos_bag_mask) + 1e-12)

  cost_list = [avg_cost1, tf.constant(0, dtype=config.data_type), avg_cost3]
  final_cost = (
          config.gamma * avg_cost1 +
          # config.alpha * avg_cost2 +
          config.beta  * avg_cost3
          )

  # ======================
  # CALCULATED REGULATIONS
  if False:
    regu_asserts = []
    # 1 for padded sentence
    flip_sent_mask = (tf.constant(1, shape=[config.batch_size, sent_num], dtype=config.data_type) - sentence_mask) # [batch_size, sent_num]
    hete_mask = tf.to_float(trgt_mask) # [batch_size]
    # hete_mask = tf.Print(hete_mask, [hete_mask], message='hete_mask')
    homo_mask = (tf.constant(1, shape=[config.batch_size], dtype=config.data_type) - hete_mask)
    # homo_mask = tf.Print(homo_mask, [homo_mask], message='hete_mask')
    regu_2ends = regulation_2ends(config, preds, max_pred, flip_sent_mask, hete_mask, homo_mask)
    # regu_middle = regulation_middle(config, preds, sentence_mask_b[b], flip_sent_mask, hete_mask, sent_num)
    hinge_loss = bag_hinge_loss(config, preds, sentence_mask, flip_sent_mask, hete_mask, sentence_trgt, sent_num)

    # FINAL REGULARIZATION FOR MIL LEARNING
    # final_regu = tf.reduce_sum(regu_2ends, name='regu')
    # final_regu = tf.reduce_sum(regu_2ends + regu_middle, name='regu')
    with tf.control_dependencies(regu_asserts): # FIXME: may well assert mask range
      final_regu = tf.reduce_sum(regu_2ends + hinge_loss, name='regu')
  else:
    final_regu = tf.constant(0, dtype=config.data_type)

  bag_pred = max_pred
  bag_cost = (final_cost)
  bag_sent_cost = sent_costs
  bag_regu = final_regu
  return bag_pred, bag_cost, bag_sent_cost, bag_regu, cost_list


def group_pick(config, sent_num,
       logits, # [batch_size, sent_num]
       preds,  # [batch_size, sent_num]
       sentence_mask, # [batch_size, sent_num]
       sentence_trgt, # [batch_size, sent_num]
       trgt, # [batch_size]
       trgt_mask, # [batch_size]
       ):
  """ Calculate predication and cost for the *bag*
  Choose the closest one to represent the bag for training now.
  """
  sent_costs = []
  cost_list = [] # Different parts/terms in the overall cost function

  mask_asserts = []
  pos_mask = sentence_trgt * sentence_mask         # [batch_size, sent_num] # positive sentences
  neg_mask = (1.0 - sentence_trgt) * sentence_mask # [batch_size, sent_num] # negative sentences
  mask_asserts.append(tf.assert_non_negative(pos_mask))
  #mask_asserts.append(tf.assert_non_negative(neg_mask))
  #mask_asserts.append(tf.assert_equal(pos_mask + neg_mask, sentence_mask,
  #    data=[pos_mask, neg_mask, sentence_mask, sentence_trgt], summarize=100))
  pos_bag_mask = trgt       # [batch_size] # positive bag
  neg_bag_mask = 1.0 - trgt # [batch_size] # negative bag
  #mask_asserts.append(tf.assert_non_negative(pos_bag_mask))
  #mask_asserts.append(tf.assert_non_negative(neg_bag_mask))
  #mask_asserts.append(tf.assert_equal(pos_bag_mask + neg_bag_mask,
  #    tf.constant(1, shape=[config.batch_size], dtype=config.data_type)))

  # ============================
  # COST #1: FOR NEGATIVE BAGS
  # MAKE THE MAXIMUM SCORE < 0.5
  cost1_asserts = []
  preds_pad2zero = preds * sentence_mask # [batch_size, sent_num]
  max_pred = tf.reduce_max(preds_pad2zero, 1, name='max_pred') # [batch_size]
  cost1 = neg_bag_mask * tf.reduce_max(tf.pack(
      [tf.constant(0, shape=[config.batch_size], dtype=config.data_type),
       config.neg_margin - (0.5 - max_pred)], axis=1), 1) # [batch_size]
  #cost1_asserts.append(tf.assert_non_negative(cost1))
  with tf.control_dependencies(cost1_asserts + mask_asserts):
    avg_cost1 = tf.reduce_sum(cost1) / (tf.reduce_sum(neg_bag_mask) + 1e-12)

  # ============================================
  # COST #2: FOR POSITIVE BAGS
  # MAKE SURE THE LARGEST IS A POSITIVE INSTANCE
  cost2_asserts = []
  max_pos_pred = tf.reduce_max(preds * pos_mask, 1) # [batch_size]
  max_neg_pred = tf.reduce_max(preds * neg_mask, 1) # [batch_size]
  cost2 = pos_bag_mask * tf.reduce_max(tf.pack(
      [tf.constant(0, shape=[config.batch_size], dtype=config.data_type),
       config.pos_neg_margin - (max_pos_pred - max_neg_pred)], axis=1), 1)
  #cost2_asserts.append(tf.assert_non_negative(cost2))
  #cost2_asserts.append(tf.assert_equal(max_neg_pred * neg_bag_mask,
  #                                     max_pred * neg_bag_mask))
  with tf.control_dependencies(cost2_asserts):
    avg_cost2 = tf.reduce_sum(cost2) / (tf.reduce_sum(pos_bag_mask) + 1e-12)

  # ===============================================================
  # COST #3: ALSO FOR POSITIVE BAGS
  # GET THE LARGEST FROM A POSITIVE SENTS; MAKE SURE IT'T ABOVE 0.5
  cost3_asserts = []
  cost3 = pos_bag_mask * tf.reduce_max(tf.pack(
      [tf.constant(0, shape=[config.batch_size], dtype=config.data_type),
       config.pos_margin - (max_pos_pred - 0.5)], axis=1), 1) # [batch_size]
  #cost3_asserts.append(tf.assert_non_negative(cost3))
  with tf.control_dependencies(cost3_asserts):
    avg_cost3 = tf.reduce_sum(cost3) / (tf.reduce_sum(pos_bag_mask) + 1e-12)

  cost_list = [avg_cost1, avg_cost2, avg_cost3]
  final_cost = (
                config.gamma * avg_cost1 +
                config.alpha * avg_cost2 +
                config.beta  * avg_cost3
                )

  final_regu = tf.constant(0, dtype=config.data_type)

  bag_pred = max_pred
  bag_cost = (final_cost)
  bag_sent_cost = sent_costs
  bag_regu = final_regu
  return bag_pred, bag_cost, bag_sent_cost, bag_regu, cost_list


def bag_hinge_loss(config, preds, sent_mask, flip_sent_mask, hete_mask,
                   sent_trgt, sent_num):
  """ HINGE LOSS:
      DEFINED AS: MAX(0, M - MIN(SENT+) - MAX(SENT-))
      THIS ONLY APPLIES TO HETE BAGS.
  """
  flip_sent_trgt = \
      tf.constant(1, shape=[config.batch_size,sent_num], dtype=config.data_type) - \
      sent_trgt
  pos_preds = preds + flip_sent_trgt + flip_sent_mask # [batch_size, sent_num]
  neg_preds = preds * flip_sent_trgt * sent_mask # [batch_size, sent_num]
  min_pos_pred = tf.reduce_min(pos_preds, 1)
  # min_pos_pred = tf.Print(min_pos_pred, [min_pos_pred], message='min_pos_pred')
  max_neg_pred = tf.reduce_max(neg_preds, 1)
  # max_neg_pred = tf.Print(max_neg_pred, [max_neg_pred], message='max_neg_pred')

  hinge_loss = hete_mask * tf.reduce_max(tf.pack(
      [tf.constant(0, shape=[config.batch_size], dtype=config.data_type),
       (0.20 - min_pos_pred + max_neg_pred)], axis=1), 1) # [batch_size]
  # hinge_loss = tf.Print(hinge_loss, [hinge_loss], message='hinge_loss', summarize=20)

  avg_hinge_loss = tf.reduce_sum(hinge_loss) / (tf.reduce_sum(hete_mask) + 1e-12)
  return avg_hinge_loss
