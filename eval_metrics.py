
import numpy as np
import pdb

# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score
from sklearn.metrics import average_precision_score
# http://scikit-learn.org/stable/modules/model_evaluation.html#label-ranking-average-precision
from sklearn.metrics import label_ranking_average_precision_score


def label_ranking_reciprocal_rank(label,  # [sent_num]
                                  preds): # [sent_num]
  """ Calcualting the reciprocal rank according to definition,
  """
  rank = np.argsort(preds)[::-1]

  #pos_rank = np.take(rank, np.where(label == 1)[0])
  #return np.mean(1.0 / pos_rank)
  
  if_find = False 
  pos = 0
  for r in rank:
      pos += 1
      if label[r] == 1:
          first_pos_r = pos
          if_find = True
          break

  assert(if_find)

  return 1.0 / first_pos_r


class EvalStats():
  def __init__(self):
    self.bag_true_pos = 0
    self.bag_true_neg = 0
    self.bag_false_pos = 0
    self.bag_false_neg = 0
    self.true_pos = 0
    self.true_neg = 0
    self.false_pos = 0
    self.false_neg = 0
    self.average_precisions = []
    self.reciprocal_ranks = []

  def reset(self):
    self.bag_true_pos = 0
    self.bag_true_neg = 0
    self.bag_false_pos = 0
    self.bag_false_neg = 0
    self.true_pos = 0
    self.true_neg = 0
    self.false_pos = 0
    self.false_neg = 0
    del self.average_precisions[:]
    del self.reciprocal_ranks[:]


  def _batch_MAP_MRR(self,
                     s_label, # [batch_size, sent_num]
                     s_preds, # [batch_size, sent_num]
                     mask):   # [batch_size, sent_num]
    """ Calcualte the Mean Average Precision and Mean Reciprocal Rank
    """
    average_precisions = []
    reciprocal_ranks = []
    for i in xrange(s_label.shape[0]): # For each question in the batch

      # Only keep those not padded
      label = np.take(s_label[i], np.where(mask[i] == 1)[0])
      preds = np.take(s_preds[i], np.where(mask[i] == 1)[0])
      assert(label.shape == preds.shape)

      # MAP only makes sense for positive bags
      try:
        assert(np.max(label) > 0)
      except AssertionError as e:
        print(s_label)
        raise e

      # TODO: is this correct???
      ap = label_ranking_average_precision_score([label], # true binary label
                                                 [preds]) # target scores
      rr = label_ranking_reciprocal_rank(label, preds)

      try: assert(not np.isnan(ap) and not np.isnan(rr))
      except: pdb.set_trace()

      average_precisions.append(ap)
      reciprocal_ranks.append(rr)
    return average_precisions, reciprocal_ranks

  def update_stats(self,
                   s_label=None, # [batch_size, sent_num] sentence level
                   s_preds=None, # [batch_size, sent_num] sentence level
                   mask=None,
                   if_pretrain=False):
    """
    Update the evaluation statistics.
      s_preds: sentence level predictions
      s_label: sentence level golden labels, same size with 's_preds'
      mask: binary mask, 1 elements indicate useful 's_preds' and 's_label' at the
            corresponding locations; while 0 elements indicates not useful, padding
            information. Size the same as 's_preds' and 's_label'
    """
    assert(s_label.shape == s_preds.shape)
    if mask is not None: assert(mask.shape == s_preds.shape)
    else: mask = np.ones(s_preds.shape)

    # pdb.set_trace()
    # MAP and MRR (metrics for ranking performance)
    # We follow the literature to only evaluate MAP and MRR for positive bags.
    pos_bag_idx = np.where(np.sum(s_label * mask, axis=1) > 0)[0]
    s_label_pos_bag = s_label[pos_bag_idx]
    s_preds_pos_bag = s_preds[pos_bag_idx]
    mask_pos_bag = mask[pos_bag_idx]

    aps, rrs = self._batch_MAP_MRR(s_label_pos_bag,
                                   s_preds_pos_bag,
                                   mask_pos_bag)
    self.average_precisions += aps
    self.reciprocal_ranks += rrs

    if if_pretrain: # all with shape [batch_size, sent_num]
      binary_preds = (s_preds > 0.5).astype(np.float32)

      correct = mask * (binary_preds == s_label).astype(np.float32)
      false = mask * (binary_preds != s_label).astype(np.float32)

      true_pos = s_label * correct
      true_neg = correct - true_pos
      false_neg = s_label * false
      false_pos = false - false_neg
      assert(((true_pos + true_neg + false_neg + false_pos) == mask).all())

    else:
      # For bag level answer triggering task, the question is answered correctly
      # if and only if the following two conditions are met:
      # (1) there's an snwer (2) the top ranked answer is the true answer
      #
      # There are several cases here:
      # (1) predict no answer, actually no answer.
      #     In this case, the bag level prediction is correct, and
      #     the sentence level prediction must also be correct, e.g. the top
      #     ranked is predicted not an answer.
      # (2) predict no answer, actually there's an answer
      #     In this case, the bag level prediction is incorrect, but
      #     the sentence level prediction may or may not be correct
      # (3) predict answer, actually no answer.
      #     In this case, the bag level prediction is incorrect, and the
      #     sentence level prediction must also be incorrect, e.g. the top
      #     ranked answer is actually not an answer.
      # (4) predict answer, actually there's an answer
      #     In this case, the bag level prediction is correct, but
      #     the sentence level prediction may or may not be correct. The
      #     prediction is correct if and only if both bag level and sentence
      #     level predictiona are correct.
      #
      # In sum, correct prediction must meet this criterias:
      # Both bag level prediction and sentence level prediction are correct, no
      # matter it's in which one of the four cases above.

      # Bag level prediction results
      binary_bag_preds = (np.max(mask * s_preds, axis=1) > 0.5).astype(np.float32)
      bag_label = np.max(s_label, axis=1) # [batch_size]
      bag_correct = (binary_bag_preds == bag_label).astype(np.float32) # [batch_size]
      bag_false = (binary_bag_preds != bag_label).astype(np.float32)   # [batch_size]

      bag_true_pos = binary_bag_preds * bag_correct
      bag_true_neg = bag_correct - bag_true_pos
      bag_false_pos = binary_bag_preds * bag_false
      bag_false_neg = bag_false - bag_false_pos
      assert(((bag_true_pos + bag_true_neg + bag_false_neg + bag_false_pos)
             == 1).all()) # Every bag should fall into one of the four cases

      # Sentence level prediction results
      one_hot = np.zeros(s_label.shape)
      one_hot[np.arange(s_label.shape[0]),
              np.argmax(mask * s_preds, axis=1)] = 1
      binary_top_preds = binary_bag_preds
      top_label = np.max(one_hot * s_label, axis=1)
      sent_correct = (binary_top_preds == top_label).astype(np.float32)
      sent_false = (binary_top_preds != top_label).astype(np.float32)

      correct = bag_correct * sent_correct # [batch_size]
      false = ((bag_false + sent_false) > 0).astype(np.float32)

      true_pos = binary_top_preds * correct
      true_neg = correct - true_pos
      false_pos = binary_top_preds * false
      false_neg = false - false_pos
      assert(((true_pos + true_neg + false_neg + false_pos) == 1).all())

    self.bag_true_pos += np.sum(bag_true_pos)
    self.bag_true_neg += np.sum(bag_true_neg)
    self.bag_false_pos += np.sum(bag_false_pos)
    self.bag_false_neg += np.sum(bag_false_neg)
    self.true_pos += np.sum(true_pos)
    self.true_neg += np.sum(true_neg)
    self.false_pos += np.sum(false_pos)
    self.false_neg += np.sum(false_neg)
    # print(correct)
    # print(binary_top_preds)
    # print(true_pos)
    # print(true_neg)
    # print(false_pos)
    # print(false_neg)
    rslt = np.stack([true_pos, true_neg, false_pos, false_neg], axis=1)
    assert((np.sum(rslt, axis=1) == 1).all()) # assert one-hot
    return np.where(rslt)[1]

  def precision(self):
    """ Return (1) answer triggering and (2) bag label prediction results
    """
    if self.true_pos == 0.0: p = 0.0
    else: p = float(self.true_pos) / \
                  float(self.true_pos + self.false_pos)
    if self.bag_true_pos == 0: b_p = 0.0
    else: b_p = float(self.bag_true_pos) / \
                    float(self.bag_true_pos + self.bag_false_pos)
    return p, b_p

  def recall(self):
    """ Return (1) answer triggering and (2) bag label prediction results
    """
    if self.true_pos == 0.0: r = 0.0
    else: r = float(self.true_pos) / \
                  float(self.true_pos + self.false_neg)
    if self.bag_true_pos == 0.0: b_r = 0.0
    else: b_r = float(self.bag_true_pos) / \
                    float(self.bag_true_pos + self.bag_false_neg)
    return r, b_r

  def f1(self):
    """ Return (1) answer triggering and (2) bag label prediction results
    """
    precision, b_precision = self.precision()
    recall, b_recall = self.recall()
    if precision == 0.0 or recall == 0.0: f1 = 0.0
    else: f1 = 2 * (precision * recall) / (precision + recall)
    if b_precision == 0.0 or b_recall == 0.0: b_f1 = 0.0
    else: b_f1 = 2 * (b_precision * b_recall) / (b_precision + b_recall)
    return f1, b_f1

  def accuracy(self):
    """ Return (1) answer triggering and (2) bag label prediction results
    """
    numerator = float(self.true_pos + self.true_neg)
    denominator = float(numerator + self.false_pos + self.false_neg)
    accu = numerator / denominator
    b_numerator = float(self.bag_true_pos + self.bag_true_neg)
    b_denominator = float(b_numerator + self.bag_false_pos + self.bag_false_neg)
    b_accu = b_numerator / b_denominator
    return accu, b_accu

  def MAP(self):
    return np.mean(self.average_precisions)

  def MRR(self):
    return np.mean(self.reciprocal_ranks)
