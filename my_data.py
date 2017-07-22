#
# The data manager.
# It will prepare feedable data for training, validation or testing.
#
################################################################################
import numpy as np
import random
import pdb

REVERSE_QUESTION = False


class RandomIndex():
  """
  Randomly generate indexes within a range, with guarantee that every index will
  be generated.
  """
  def __init__(self, num_b):
    self._num_b = num_b
    self._index_queue_b = []
    for b in xrange(len(num_b)):
      self._index_queue_b.append([])


  def sample(self, b, sample_num):
    """
    """
    while len(self._index_queue_b[b]) < sample_num:
      new_idx_list = range(self._num_b[b])
      random.shuffle(new_idx_list)
      self._index_queue_b[b] += new_idx_list
    sample_indexes = self._index_queue_b[b][:sample_num]
    assert(len(sample_indexes) == sample_num)
    del self._index_queue_b[b][:sample_num]

    return sample_indexes



class MyData():
  def __init__(self,
               train_tuple_b,
               valid_tuple_b=None,
               test_tuple_b=None,
               word2id=None,
               word_embeddings=None,
               if_balance=None):
    """ Build data manager
    # Each bucket is indicated by three numbers:
    # (1) sentence_number, (2) question_length, (3) sentence_length

    # Each data tuple has four elements:
    # (1) questions (2) sentences (3) bag labels (4) sentence labels
    """
    print('----- ----- loading training data ----- -----')
    if train_tuple_b is not None:
      # if if_balance: # Down sample the data to get a pos-neg balance
      #   assert(if_balance == 'down' or if_balance == 'over')
      #   if if_balance == 'over':
      #     train_tuple_b = self._balance(train_tuple_b, 'over')
      #   train_tuple_b = self._balance(train_tuple_b, 'down')
      #   self.train_tuple_b = self._balance(train_tuple_b, 'down') # just for checking

      train_tuple_b, self.train_buckets, self.train_bucket_sizes, bag_labels = \
          self._parse(train_tuple_b)
      self.train_bucket_pos_sizes, self.train_bucket_neg_sizes, \
          self.train_bucket_pos_idx, self.train_bucket_neg_idx = \
          self._parse_polar(train_tuple_b)

      train_total_sizes = float(sum(self.train_bucket_sizes))
      self.train_bucket_scale = \
          [sum(self.train_bucket_sizes[:i+1]) / train_total_sizes
           for i in xrange(len(self.train_bucket_sizes))]

      self.train_questions_b = train_tuple_b[:, 0]
      self.train_sentences_b = train_tuple_b[:, 1]
      self.train_sentence_labels_b = train_tuple_b[:, 2]
      self.train_question_feats_b = train_tuple_b[:, 3]
      self.train_sentence_feats_b = train_tuple_b[:, 4]
      self.train_labels_b = bag_labels
      self.train_question_padding_mask_b = \
          [(q != 0).astype(np.float32) for q in self.train_questions_b]
      self.train_sentences_padding_mask_b = \
          [(s != 0).astype(np.float32) for s in self.train_sentences_b]
      if REVERSE_QUESTION:
        self._reverse_question_tokens(self.train_questions_b,
                                      self.train_question_padding_mask_b)

      for b in xrange(len(self.train_bucket_sizes)):
        assert(self.train_bucket_pos_sizes[b] + self.train_bucket_neg_sizes[b] ==
               self.train_bucket_sizes[b])
        print("bucket %d: pos bag #: %d\tneg bag #: %d\ttotal #: %d" %
              (b, self.train_bucket_pos_sizes[b], self.train_bucket_neg_sizes[b],
               self.train_bucket_sizes[b]))
      self._train_pos_index_sampler = RandomIndex(self.train_bucket_pos_sizes)
      self._train_neg_index_sampler = RandomIndex(self.train_bucket_neg_sizes)

    print('----- ----- loading validation data ----- -----')
    if valid_tuple_b is not None:
      if if_balance: # Down sample the data to get a pos-neg balance
        assert(if_balance == 'down' or if_balance == 'over')
        if if_balance == 'over':
          valid_tuple_b = self._balance(valid_tuple_b, 'over')
        valid_tuple_b = self._balance(valid_tuple_b, 'down')
        self.valid_tuple_b = self._balance(valid_tuple_b, 'down') # just for checking

      valid_tuple_b, self.valid_buckets, self.valid_bucket_sizes, bag_labels = \
          self._parse(valid_tuple_b)
      self.valid_bucket_pos_sizes, self.valid_bucket_neg_sizes, \
          self.valid_bucket_pos_idx, self.valid_bucket_neg_idx = \
          self._parse_polar(valid_tuple_b)

      assert(self.valid_buckets == self.train_buckets)
      self.valid_questions_b = valid_tuple_b[:, 0]
      self.valid_sentences_b = valid_tuple_b[:, 1]
      self.valid_sentence_labels_b = valid_tuple_b[:, 2]
      self.valid_question_feats_b = valid_tuple_b[:, 3]
      self.valid_sentence_feats_b = valid_tuple_b[:, 4]
      self.valid_labels_b = bag_labels
      if if_balance:
        for bl in bag_labels: assert(np.sum(bl) == len(bl)/2) # has been balanced
      self.valid_question_padding_mask_b = \
          [(q != 0).astype(np.float32) for q in self.valid_questions_b]
      self.valid_sentences_padding_mask_b = \
          [(s != 0).astype(np.float32) for s in self.valid_sentences_b]
      if REVERSE_QUESTION:
        self._reverse_question_tokens(self.valid_questions_b,
                                      self.valid_question_padding_mask_b)

      for b in xrange(len(self.valid_bucket_sizes)):
        assert(self.valid_bucket_pos_sizes[b] + self.valid_bucket_neg_sizes[b] ==
               self.valid_bucket_sizes[b])
        print("bucket %d: pos bag #: %d\tneg bag #: %d\ttotal #: %d" %
              (b, self.valid_bucket_pos_sizes[b], self.valid_bucket_neg_sizes[b],
               self.valid_bucket_sizes[b]))

    else:
      print("Validation data not available")

    print('----- ----- loading testing data ----- -----')
    if test_tuple_b is not None:

      if if_balance: # Down sample the data to get a pos-neg balance
        assert(if_balance == 'down' or if_balance == 'over')
        if if_balance == 'over':
          test_tuple_b = self._balance(test_tuple_b, 'over')
        test_tuple_b = self._balance(test_tuple_b, 'down')
        self.test_tuple_b = self._balance(test_tuple_b, 'down') # just for checking

      test_tuple_b, self.test_buckets, self.test_bucket_sizes, bag_labels = \
          self._parse(test_tuple_b)
      self.test_bucket_pos_sizes, self.test_bucket_neg_sizes, \
          self.test_bucket_pos_idx, self.test_bucket_neg_idx = \
          self._parse_polar(test_tuple_b)

      assert(self.test_buckets == self.train_buckets)
      self.test_questions_b = test_tuple_b[:, 0]
      self.test_sentences_b = test_tuple_b[:, 1]
      self.test_sentence_labels_b = test_tuple_b[:, 2]
      self.test_question_feats_b = test_tuple_b[:, 3]
      self.test_sentence_feats_b = test_tuple_b[:, 4]
      self.test_labels_b = bag_labels
      if if_balance:
        for bl in bag_labels: assert(np.sum(bl) == len(bl)/2) # has been balanced
      self.test_question_padding_mask_b = \
          [(q != 0).astype(np.float32) for q in self.test_questions_b]
      self.test_sentences_padding_mask_b = \
          [(s != 0).astype(np.float32) for s in self.test_sentences_b]
      if REVERSE_QUESTION:
        self._reverse_question_tokens(self.test_questions_b,
                                      self.test_question_padding_mask_b)

      for b in xrange(len(self.test_bucket_sizes)):
        assert(self.test_bucket_pos_sizes[b] + self.test_bucket_neg_sizes[b] ==
               self.test_bucket_sizes[b])
        print("bucket %d: pos bag #: %d\tneg bag #: %d\ttotal #: %d" %
              (b, self.test_bucket_pos_sizes[b], self.test_bucket_neg_sizes[b],
               self.test_bucket_sizes[b]))

    else:
      print("Testing data not available")

    if word2id is not None:
      self.word2id = word2id
      self.id2word = {}
      for key, value in word2id.iteritems():
        assert(value not in word2id)
        self.id2word[value] = key
    else:
      print("Word-Id mapping not available")

    if word_embeddings is not None:
      self.word_embeddings = word_embeddings
    else:
      print("Word embedding not available")


  def _balance(self, tuple_b, mode):
    """ Down or over sample to balance
    """
    assert(mode == 'down' or mode == 'over')
    new_tuple_b = []
    for bid, (questions, sentences, sent_labels) in enumerate(tuple_b):
      questions = np.array(questions)
      sentences = np.array(sentences)
      sent_labels = np.array(sent_labels)
      bag_labels = sent_labels.max(axis=1)
      num_pos = np.sum(bag_labels)
      num_neg = len(bag_labels) - num_pos
      print('Balancing bags %d: %d positive; %d negative' %
            (bid, num_pos, num_neg))
      if mode == 'down':
        if num_pos > num_neg:
          print('down sample positive')
          pos_idx = np.where(bag_labels == 1)[0]
          del_idx = np.random.choice(pos_idx, num_pos-num_neg, replace=False)
          keep_idx = np.delete(np.arange(len(bag_labels)), del_idx)
          questions = questions[keep_idx]
          sentences = sentences[keep_idx]
          sent_labels = sent_labels[keep_idx]
        elif num_pos < num_neg:
          print('down sample negative')
          neg_idx = np.where(bag_labels == 0)[0]
          del_idx = np.random.choice(neg_idx, num_neg-num_pos, replace=False)
          keep_idx = np.delete(np.arange(len(bag_labels)), del_idx)
          questions = questions[keep_idx]
          sentences = sentences[keep_idx]
          sent_labels = sent_labels[keep_idx]
        else:
          print('balanced already')
      else: # mode is 'over'
        if num_pos > num_neg:
          raise NotImplementedError("Over sampling negative bags hasn't been implemented.")
        elif num_pos < num_neg:
          print('over sample positive')
          # The strategy here is to choose positive bags that has as many sentences as possible
          # and those with more than one positive sentences.
          sent_pad_mask = (np.sum((sentences != 0).astype(int), axis=2) != 0).astype(int)
          n_sent = np.sum(sent_pad_mask, axis=1) # none padding sentences in each bag
          n_pos = np.sum(sent_labels, axis=1) # positive sentences in each bag
          n_sent_max = np.max(n_sent)
          splitted_id = set([])
          n_sample = 0
          new_qs = [] # questions of new sampled bags
          new_ss = [] # sentences of new sampled bags
          new_sl = [] # sentence labels of new sampled bags
          while True:
            randids = np.random.permutation(np.intersect1d(
                np.where(n_sent == n_sent_max)[0], np.where(n_pos > 1)[0]))
            # Split positive bags, each contains only one positive instance
            for idx in randids:
              bag_size = len(sent_labels[idx])
              pos_idx = np.random.permutation(np.where(sent_labels[idx] == 1)[0])
              neg_idx = np.where(sent_labels[idx] == 0)[0]
              assert(pos_idx.size > 1)
              for i in xrange(len(pos_idx)):
                bg_sent_idx = np.random.permutation(
                    np.concatenate((pos_idx[[i]], neg_idx), axis=0))
                new_s = np.pad(sentences[idx][bg_sent_idx],
                               pad_width=((0,bag_size-len(bg_sent_idx)),(0,0)),
                               mode='constant', constant_values=0)
                assert(new_s.shape == sentences[idx].shape)
                new_s_label = np.pad(sent_labels[idx][bg_sent_idx],
                                     pad_width=((0,bag_size-len(bg_sent_idx))),
                                     mode='constant', constant_values=0)
                assert(new_s_label.shape == sent_labels[idx].shape)
                if n_sample < (num_neg - num_pos):
                  new_qs.append(questions[idx])
                  new_ss.append(new_s)
                  new_sl.append(new_s_label)
                  n_sample += 1
                else:
                  print("Got enough samples")
                  break
              if n_sample >= (num_neg - num_pos): break
            if n_sent_max == 0:
              print("Splitted all the sentences: got %d more" % len(new_qs))
              assert(len(randids) == 0)
              break
            n_sent_max -= 1 # used all the max sentence bags, go to the next level
          # Merge to original data
          new_qs = np.array(new_qs)
          new_ss = np.array(new_ss)
          new_sl = np.array(new_sl)
          questions = np.concatenate([questions, new_qs], axis=0)
          sentences = np.concatenate([sentences, new_ss], axis=0)
          sent_labels = np.concatenate([sent_labels, new_sl], axis=0)
        else:
          print('balanced already')
        #
      new_tuple_b.append([questions, sentences, sent_labels])
    return np.array(new_tuple_b)


  def _parse(self, tuple_b):
    """ Pass train_tuple_b, valid_tuple_b OR test_tuple_b
    """
    buckets = []
    bucket_sizes = []
    bag_labels_b = []
    new_tuple_b = []
    for questions, sentences, sent_labels, ques_feats, sent_feats in tuple_b:
      questions = np.array(questions) # [bucket_size, ques_length]
      sentences = np.array(sentences) # [bucket_size, sent_num, sent_length]
      sent_labels = np.array(sent_labels) # [bucket_size, sent_num]
      # Only one feature: question length
      ques_feats = np.array(ques_feats) # [bucket_size, 1]
      # Three features: sentence length + two word overlapping features 
      sent_feats = np.array(sent_feats) # [bucket_size, sent_num, 3]

      # Parse label data
      assert(len(sent_labels.shape) == 2)
      bucket_size = sent_labels.shape[0]
      sentence_number = sent_labels.shape[1]
      bag_labels = sent_labels.max(axis=1) # [bucket_size]
      # Parse question data
      assert(len(questions.shape) == 2)
      assert(questions.shape[0] == bucket_size)
      quesiton_length = questions.shape[1]
      # Parse sentences data
      assert(len(sentences.shape) == 3)
      assert(sentences.shape[0] == bucket_size)
      assert(sentences.shape[1] == sentence_number)
      sentence_length = sentences.shape[2]
      # Parse question feature data
      assert(len(ques_feats.shape) == 2)
      assert(ques_feats.shape[0] == bucket_size)
      assert(ques_feats.shape[1] == 1)
      # Parse sentence feature data
      assert(len(sent_feats.shape) == 3)
      assert(sent_feats.shape[0] == bucket_size)
      assert(sent_feats.shape[1] == sentence_number)
      assert(sent_feats.shape[2] == 3)

      bucket_sizes.append(bucket_size)
      buckets.append([sentence_number, quesiton_length, sentence_length])
      bag_labels_b.append(bag_labels)
      new_tuple_b.append([questions, sentences, sent_labels, ques_feats, sent_feats])

    return np.array(new_tuple_b), buckets, bucket_sizes, np.array(bag_labels_b)


  def _parse_polar(self, tuple_b):
    """
    """
    bucket_pos_size = []
    bucket_neg_size = []
    bucket_pos_idx = []
    bucket_neg_idx = []

    for questions, sentences, sent_labels, ques_feats, sent_feats in tuple_b:
      label = np.max(sent_labels, axis=1)
      pos_idx = np.where(label == 1)[0]
      neg_idx = np.where(label == 0)[0]
      n_pos = pos_idx.size
      n_neg = neg_idx.size
      assert(n_pos + n_neg == label.size)
      assert(len(set(pos_idx) & set(neg_idx)) == 0)

      bucket_pos_size.append(n_pos)
      bucket_neg_size.append(n_neg)
      bucket_pos_idx.append(pos_idx)
      bucket_neg_idx.append(neg_idx)

    return bucket_pos_size, bucket_neg_size, bucket_pos_idx, bucket_neg_idx


  def sample_train_buckets(self):
    """ Sample a bucket according to the buckets sizes
    """
    random_number_01 = np.random.random_sample()
    bucket_id = min([i for i in xrange(len(self.train_bucket_scale))
                     if self.train_bucket_scale[i] > random_number_01])
    return bucket_id


  def _reverse_question_tokens(self,
                               question_b, # list of [batch_size, q_len]
                               question_padding_mask_b): # same shape as 'question_b'
    """ Reverse question tokens, while leaving padding still at the end
    """
    assert(len(question_b) == len(question_padding_mask_b))
    for b in xrange(len(question_b)):
      question = question_b[b]
      question_padding_mask = question_padding_mask_b[b]
      assert(question.shape == question_padding_mask.shape)
      for i in xrange(question.shape[0]):
        non_pad_indices = np.where(question_padding_mask[i] == 1)[0]
        qtemp = np.take(question[i], non_pad_indices)
        question[i][non_pad_indices] = qtemp[::-1]


  def get_train_batch(self, bid, batch_size,
                      pretrain=False,
                      if_embed=False):
    """ Random get a batch for training.
        But, makes sure that the overall positive and negative training sample
        are the same.
    """
    if bid >= len(self.train_buckets):
      raise ValueError("bid = %d too large" % bid)
    bucket_pos_size = self.train_bucket_pos_sizes[bid]
    bucket_neg_size = self.train_bucket_neg_sizes[bid]

    if pretrain:
      # Pretrain with only positive bags, which helps balance the pos and neg
      # traing data.
      n_pos = batch_size
    else:
    #   n_pos = random.randrange(batch_size + 1)
      n_pos = np.random.binomial(batch_size, 0.5)

    # GET POSITIVE BAG INDEXES
    rand_pos_ids = self._train_pos_index_sampler.sample(bid, n_pos)
    # if n_pos < bucket_pos_size:
    #   rand_pos_ids = random.sample(xrange(bucket_pos_size), n_pos)
    # else:
    #   rand_pos_ids = []
    #   for time in xrange(n_pos):
    #     rand_pos_ids += random.sample(xrange(bucket_pos_size), 1)

    # GET NEGATIVE BAGS INDEXES
    n_neg = batch_size - n_pos
    rand_neg_ids = self._train_neg_index_sampler.sample(bid, n_neg)
    # if n_neg < bucket_neg_size:
    #   rand_neg_ids = random.sample(xrange(bucket_neg_size), n_neg)
    # else:
    #   rand_neg_ids = []
    #   for time in xrange(n_neg):
    #     rand_neg_ids += random.sample(xrange(bucket_neg_size), 1)

    # GET BAG INDEXES
    assert(bucket_pos_size == len(self.train_bucket_pos_idx[bid]))
    assert(bucket_neg_size == len(self.train_bucket_neg_idx[bid]))
    rand_ids = \
        np.concatenate([self.train_bucket_pos_idx[bid][rand_pos_ids],
                        self.train_bucket_neg_idx[bid][rand_neg_ids]],
                       axis=0)
    np.random.shuffle(rand_ids)
    assert(len(rand_ids) == batch_size)

    q = self.train_questions_b[bid][rand_ids] # [batch_size, q_len]
    s = self.train_sentences_b[bid][rand_ids] # [batch_size, sent_num, s_len]
    q_feats = self.train_question_feats_b[bid][rand_ids] # [batch_size, 1]
    s_feats = self.train_sentence_feats_b[bid][rand_ids] # [batch_size, sent_num, 3]

    s_label = self.train_sentence_labels_b[bid][rand_ids] # [batch_size, sent_num]
    label = self.train_labels_b[bid][rand_ids] # [batch_size]
    q_w_mask = self.train_question_padding_mask_b[bid][rand_ids] # [batch_size, q_len]
    s_w_mask = self.train_sentences_padding_mask_b[bid][rand_ids] # [batch_size, sent_num, s_len]
    s_mask = s_w_mask.any(axis=2).astype(np.float32) # [batch_size, sent_num]
    # print s_w_mask
    # print s_mask
    try:
      assert(q.shape == q_w_mask.shape)
      assert(q.shape == q_w_mask.shape)
      assert(s_label.shape == s_mask.shape)
      assert(q_feats.shape == (q.shape[0], 1))
      assert(s_feats.shape == (s.shape[0], s.shape[1], 3))
    except AssertionError as e:
      print(e)
      pdb.set_trace()
    return q, s, label, s_label, q_w_mask, s_w_mask, s_mask, q_feats, s_feats


  def get_test(self, bid,
               if_test=False,
               if_embed=False):
    """ Generate a validation OR test data sample, (batch = 1)
    """
    if if_test:
      if bid >= len(self.test_buckets):
        raise ValueError("bid = %d too large" % bid)
      q = self.test_questions_b[bid]
      s = self.test_sentences_b[bid]
      q_w_mask = self.test_question_padding_mask_b[bid]
      s_w_mask = self.test_sentences_padding_mask_b[bid]
      s_mask = s_w_mask.any(axis=2).astype(np.float32)
      bag_labels = self.test_labels_b[bid]
      s_labels = self.test_sentence_labels_b[bid]
      bucket_size = self.test_bucket_sizes[bid]
      q_feats = self.test_question_feats_b[bid]
      s_feats = self.test_sentence_feats_b[bid]
    else:
      if bid >= len(self.valid_buckets):
        raise ValueError("bid = %d too large" % bid)
      q = self.valid_questions_b[bid]
      s = self.valid_sentences_b[bid]
      q_w_mask = self.valid_question_padding_mask_b[bid]
      s_w_mask = self.valid_sentences_padding_mask_b[bid]
      s_mask = s_w_mask.any(axis=2).astype(np.float32)
      bag_labels = self.valid_labels_b[bid]
      s_labels = self.valid_sentence_labels_b[bid]
      bucket_size = self.valid_bucket_sizes[bid]
      q_feats = self.valid_question_feats_b[bid]
      s_feats = self.valid_sentence_feats_b[bid]
        
    # print s_w_mask
    # print s_mask
    for i in range(bucket_size):
      yield q[[i]], s[[i]], bag_labels[[i]], s_labels[[i]], \
          q_w_mask[[i]], s_w_mask[[i]], s_mask[[i]], \
          q_feats[[i]], s_feats[[i]]


  def get_test_batch(self, bid, batch_size,
                     if_test=False,
                     if_embed=False):
    """ Generate a validation OR test data sample, (batch = 1)
    """
    if if_test:
      if bid >= len(self.test_buckets):
        raise ValueError("bid = %d too large" % bid)
      q = self.test_questions_b[bid]
      s = self.test_sentences_b[bid]
      q_w_mask = self.test_question_padding_mask_b[bid]
      s_w_mask = self.test_sentences_padding_mask_b[bid]
      s_mask = s_w_mask.any(axis=2).astype(np.float32)
      bag_labels = self.test_labels_b[bid]
      s_labels = self.test_sentence_labels_b[bid]
      bucket_size = self.test_bucket_sizes[bid]
      q_feats = self.test_question_feats_b[bid]
      s_feats = self.test_sentence_feats_b[bid]
    else:
      if bid >= len(self.valid_buckets):
        raise ValueError("bid = %d too large" % bid)
      q = self.valid_questions_b[bid]
      s = self.valid_sentences_b[bid]
      q_w_mask = self.valid_question_padding_mask_b[bid]
      s_w_mask = self.valid_sentences_padding_mask_b[bid]
      s_mask = s_w_mask.any(axis=2).astype(np.float32)
      bag_labels = self.valid_labels_b[bid]
      s_labels = self.valid_sentence_labels_b[bid]
      bucket_size = self.valid_bucket_sizes[bid]
      q_feats = self.valid_question_feats_b[bid]
      s_feats = self.valid_sentence_feats_b[bid]

    if bucket_size < batch_size:
      raise ValueError("Use a smaller batch size")
    i_range = range(0, bucket_size, batch_size)
    overlap = i_range[-1] + batch_size - bucket_size
    assert(overlap >= 0)
    if overlap > 0:
      i_range[-1] = bucket_size - batch_size

    for i in i_range:
      ids = range(i, i + batch_size)
      o = overlap if i == i_range[-1] else 0
      yield q[ids], s[ids], bag_labels[ids], s_labels[ids], \
          q_w_mask[ids], s_w_mask[ids], s_mask[ids], \
          q_feats[ids], s_feats[ids], o


if __name__ == "__main__":
    from prep_wikiqa_data import prep_data, BUCKETS, WordVecs

    train_tuple_b, valid_tuple_b, test_tuple_b, word2id, word_embeddings = \
        prep_data()

    pdb.set_trace()
    mydata = MyData(train_tuple_b, valid_tuple_b, test_tuple_b, word2id)

    bid = 0
    batch_size = 7
    mydata.get_train_batch(bid, batch_size)
    pdb.set_trace()

    for question, sentences, label, sent_labels, q_mask, s_w_mask, s_mask, q_feats, s_feats in \
        mydata.get_test(bid, if_test=False):
      pass
    pdb.set_trace()
