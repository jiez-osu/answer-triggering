#
# This script handles the preparation of the WikiQA dataset.
# It will be used by the tensorflow model.
################################################################################

import cPickle as pickle
import pdb
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Each bucket is indicated by three numbers:
# (1) sentence_number, (2) question_length, (3) sentence_length
BUCKETS = [[5, 40, 40], [15, 40, 40], [30, 40, 40]]
# BUCKETS = [[5, 25, 50], [10, 25, 50], [10, 25, 50], [10, 25, 100],
#            [20, 25, 50], [20, 25, 100], [30, 25, 50], [30, 25, 100]]

# Bucket collaps means ignore the first dimension of buckets:
# sentence_number.
# This will be used in the bag sampling technique defined
# "my_data_plus.py".
# It coresponds to on with the "if_pad_token_only" option in
# function "cast_data_to_buckets".
BUCKETS_COLLAPSED = [[40, 40]]


# Borrowed from "process_data.py" from the "WikiQACodePackage",
# from http://research.microsoft.com/en-US/downloads/a5c91569-d291-4450-aaab-5bce7995fe0c/default.aspx
class WordVecs(object):
    """
    precompute embeddings for word/feature/tweet etc.
    """
    def __init__(self, fname, vocab, binary=1, has_header=False):
        if binary == 1:
            word_vecs = self.load_bin_vec(fname, vocab)
        else:
            word_vecs = self.load_txt_vec(fname, vocab, has_header)
        self.k = len(word_vecs.values()[0])
        self.add_unknown_words(word_vecs, vocab, k=self.k)
        self.W, self.word_idx_map = self.get_W(word_vecs, k=self.k)

    def get_W(self, word_vecs, k=300):
        """
        Get word matrix. W[i] is the vector for word indexed by i
        """
        vocab_size = len(word_vecs)
        word_idx_map = dict()
        W = np.zeros(shape=(vocab_size+1, k))
        W[0] = np.zeros(k)
        i = 1
        for word in word_vecs:
            W[i] = word_vecs[word]
            word_idx_map[word] = i
            i += 1
        return W, word_idx_map

    def load_bin_vec(self, fname, vocab):
        """
        Loads 300x1 word vecs from Google (Mikolov) word2vec
        """
        word_vecs = {}
        with open(fname, "rb") as f:
            header = f.readline()
            vocab_size, layer1_size = map(int, header.split())
            binary_len = np.dtype('float32').itemsize * layer1_size
            for line in xrange(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch)
                if word in vocab:
                   word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
                else:
                    f.read(binary_len)
        logger.info("num words already in word2vec: " + str(len(word_vecs)))
        return word_vecs

    def load_txt_vec(self, fname, vocab, has_header=False):
        """
        Loads 50x1 word vecs from sentiment word embeddings (Tang et al., 2014)
        """
        word_vecs = {}
        pos = 0
        with open(fname, "rb") as f:
            if has_header: header = f.readline()
            for line in f:
                parts = line.strip().split()
                word = parts[0]
                if word in vocab:
                   word_vecs[word] = np.asarray(map(float, parts[1:]))
                pos += 1
        logger.info("num words already in word2vec: " + str(len(word_vecs)))
        return word_vecs

    def add_unknown_words(self, word_vecs, vocab, min_df=1, k=300):
        """
        For words that occur in at least min_df documents, create a separate word vector.
        0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
        """
        for word in vocab:
            if word not in word_vecs and vocab[word] >= min_df:
                #print word
                word_vecs[word] = np.random.uniform(-0.25,0.25,k)


# Borrowed from "qa_cnn.py" from the "WikiQACodePackage",from:
# http://research.microsoft.com/en-US/downloads/a5c91569-d291-4450-aaab-5bce7995fe0c/default.aspx
# Slightly changed since LSTM doesn't need pad on the margin but CNN does.
# Also not necessarily truncate or pad sentences.
def get_idx_from_sent(sent, word_idx_map, max_l=None):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    words = sent.split()
    for i, word in enumerate(words):
        if i >= max_l: break
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l:
        x.append(0)
    return x, len(words)

def visualize_distribution(data):
    """ Visualize bucket distribution, a distribution over:
    (sentence_number, question_length, sentence_length)"""
    bucket_distribution = []
    for qid, value in data.iteritems():
      q_length = len(value['question'])
      s_number = len(value['sentences'])
      s_length = max([len(sent) for sent in value['sentences'].itervalues()])
      bucket_distribution.append([s_number, q_length, s_length])

    # pdb.set_trace()
    distribution = np.transpose(np.array(bucket_distribution))
    kde = stats.gaussian_kde(distribution)
    density = kde(distribution)

    idx = density.argsort()
    x = distribution[0, idx]
    y = distribution[1, idx]
    z = distribution[2, idx]
    density = density[idx]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=density)
    ax.set_xlabel('sentence number')
    ax.set_ylabel('question length')
    ax.set_zlabel('sentence length')
    plt.show()

def make_data(revs, word_idx_map, max_l=None, val_test_splits=[2,3]):
    """
    Transforms sentences.

    train/val/test:
    --- dict using "qid" as key
    --- value: "question": list of intergers representing question
    --- value: "sentences":
    --- --- dict using "aid" as key
    --- --- value: list of intergers representing sentence
    --- value: "sentence_labels":
    --- --- dict using "aid" as key
    --- --- value: binary number representing if this sentence is an answer
    """
    # Formulate data for answer triggering task
    train, val, test = {}, {}, {}
    val_split, test_split = val_test_splits
    actual_question_max_l, actual_sentence_max_l = 0, 0
    for rev in revs:
        if rev["split"]==1: data_split = train
        elif rev["split"]==val_split: data_split = val
        elif rev["split"]==test_split: data_split = test
        else: raise ValueError("Unknown data split: %s" % rev["split"])
        qid, aid = rev['qid'], rev['aid']

        # Add question sentence
        if qid not in data_split:
            data_split[qid] = {}
            question, length = get_idx_from_sent(rev['question'], word_idx_map,
                                                 max_l)
            actual_question_max_l = max(length, actual_question_max_l)
            data_split[qid]['question'] = question
            data_split[qid]['question_feats'] = [rev['num_words_q']]
            data_split[qid]['sentences'] = {}
            data_split[qid]['sentence_feats'] = {}
            data_split[qid]['sentence_labels'] = {}
        # else: # Just to make sure
        #     question, length = get_idx_from_sent(rev['question'], word_idx_map, max_l)
        #     # print question
        #     # print data_split[qid]['question']
        #     assert(question == data_split[qid]['question'])

        # Add answer sentence
        assert(aid not in data_split[qid])
        sent, length = get_idx_from_sent(rev['answer'], word_idx_map, max_l)
        actual_sentence_max_l = max(length, actual_sentence_max_l)
        data_split[qid]['sentences'][aid] = sent
        data_split[qid]['sentence_feats'][aid] = [rev['num_words_a']] + rev['features']
        data_split[qid]['sentence_labels'][aid] = rev['y']
    print('Question length: max %d' % actual_question_max_l)
    print('Sentence length: max %d' % actual_sentence_max_l)

    # Check data
    # 1. make sure qid is consecutive
    for data_split in [train, val, test]:
        qids = data_split.keys()
        qids.sort()
        assert(qids == range(min(qids), max(qids)+1))
    # 2. make sure aid for each question is consecutive
    for data_split in [train, val, test]:
        max_num_sentence = 0
        min_num_sentence = 9999
        for qid, value in data_split.iteritems():
            num_sentences = len(value['sentence_labels'])
            max_num_sentence = max(num_sentences, max_num_sentence)
            min_num_sentence = min(num_sentences, min_num_sentence)
            aids_1 = value['sentences'].keys()
            aids_1.sort()
            aids_2 = value['sentence_labels'].keys()
            aids_2.sort()
            assert(aids_1 == aids_2)
            assert(aids_1 == range(min(aids_1), max(aids_2)+1))
        print('Sentence number: max %d; min %d' %
              (max_num_sentence, min_num_sentence))

    return [train, val, test]


def pad_token_only(m_buckets_collapsed, q, s, l):
    """ Pad each sample into a bucket, but only pad short sentences.
    This means do not add all paded sentences.
    This also means that the first dimension of buckets is useless.
        NOTE: use index 0 to pad.
    """
    assert(s)
    assert(len(s) == len(l))
    fit = False
    for bid, bucket in enumerate(m_buckets_collapsed):
        if len(q) <= bucket[0] and \
           len(s[0]) <= bucket[1]:
            for i in range(bucket[0] - len(q)):
                q.append(0)
            assert(len(q) == bucket[0])
            for i in range(len(s)):
                for j in range(bucket[1] - len(s[i])):
                    s[i].append(0)
                assert(len(s[i]) == bucket[1])
            fit = True
            break
    if not fit:
        print("(%d, %d, %d) doesn't fit into any bucket!" %
              (len(s), len(q), len(s[0])))
        return None
    else:
        return bid, np.array(q), np.array(s), np.array(l)


def pad(m_buckets, q, s, l, qf, sf):
    """ Pad each sample q + sents into a bucket.
        NOTE: use index 0 to pad.
    """
    assert(s)
    assert(len(s) == len(l))
    fit = False
    for bid, bucket in enumerate(m_buckets):
        if len(s) <= bucket[0] and \
           len(q) <= bucket[1] and \
           len(s[0]) <= bucket[2]:
            # Add pad sentence placeholders
            for i in range(bucket[0] - len(s)):
                s.append([])       # all-padding sentence place taker
                l.append(0)        # all_padding sentence's label is 0
                sf.append([0,0,0]) # all-padding sentence's features are zeros
            assert(len(s) == bucket[0])
            assert(len(l) == bucket[0])
            assert(len(sf) == bucket[0])
            # Add pad question tokens 
            for i in range(bucket[1] - len(q)):
                q.append(0)
            assert(len(q) == bucket[1])
            # Add pad sentence tokens
            for i in range(bucket[0]):
                for j in range(bucket[2] - len(s[i])):
                    s[i].append(0)
                assert(len(s[i]) == bucket[2])
            fit = True
            break
    if not fit:
        print("(%d, %d, %d) doesn't fit into any bucket!" %
              (len(s), len(q), len(s[0])))
        return None
    else:
        return bid, np.array(q), np.array(s), np.array(l), np.array(qf), np.array(sf)


def cast_data_to_buckets(data, m_buckets, if_pad_token_only=False):
    """ Fit data into buckets, for feeding into NN model
        data_tuple_b:
            - 1st dimension idx: bucket index
            - 2nd dimension idx:
                0 : all questions
                1 : all sentences
                2 : all labels (corresponding to sentences)
                3 : all question features
                4 : all sentence features
    """
    data_tuple_b = [[[], # question list
                     [], # sentences list
                     [], # sentence label list
                     [], # question feature list
                     []] # sentence feature list
                    for b_id in xrange(len(m_buckets))]

    for qid, value in data.iteritems():
        q = value['question']
        qf = value['question_feats']
        s, sf, l = [], [], []
        assert(value['sentences'].keys() == value['sentence_labels'].keys())
        assert(value['sentences'].keys() == value['sentence_feats'].keys())
        for aid in value['sentences'].iterkeys():
            s.append(value['sentences'][aid])
            sf.append(value['sentence_feats'][aid])
            l.append(value['sentence_labels'][aid])

        # Padding
        if not if_pad_token_only:
            b_id, question, sentences, label, q_feats, s_feats = \
                    pad(m_buckets, q, s, l, qf, sf)
        else:
            #FIXME: hasn't updated to support question features "qf" and 
            # sentence features "sf"
            b_id, question, sentences, label = \
                    pad_token_only(m_buckets, q, s, l)

        data_tuple_b[b_id][0].append(question)
        data_tuple_b[b_id][1].append(sentences)
        data_tuple_b[b_id][2].append(label)
        data_tuple_b[b_id][3].append(q_feats)
        data_tuple_b[b_id][4].append(s_feats)

    # NOTE: this has been changed since otherwise the return will generate
    # errors, when using collapsed buckets.
    # for b_id in xrange(len(data_tuple_b)):
    #     data_tuple_b[b_id][0] = np.array(data_tuple_b[b_id][0])
    #     data_tuple_b[b_id][1] = np.array(data_tuple_b[b_id][1])
    #     data_tuple_b[b_id][2] = np.array(data_tuple_b[b_id][2])

    return np.array(data_tuple_b)


def find_similar_words(wordvecs):
    """ Use loaded word embeddings to find out the most similar words in the
    embedded vector space.
    """
    from sklearn.metrics import pairwise_distances
    from scipy.spatial.distance import cosine
    pairwise_sim_mat = 1 - pairwise_distances(wordvecs.W[1:],
                                              metric='cosine',
                                              # metric='euclidean',
                                              )

    id2word = {}
    for key, value in wordvecs.word_idx_map.iteritems():
        assert(value not in id2word)
        id2word[value] = key
    while True:
        word = raw_input("Enter a word ('STOP' to quit): ")
        if word == 'STOP': break
        try:
            w_id = wordvecs.word_idx_map[word]
        except KeyError:
            print '%s not in the vocabulary.' % word
        sim_w_id  = pairwise_sim_mat[w_id-1].argsort()[-10:][::-1]
        for i in sim_w_id:
            print id2word[i+1],
        print ''


def prep_data():
    """
    The HALF preprocessed data "wiki_cnn.pkl" is generated by
    python script "process_data.py" in the "WikiQACodePackage",
    specified by path "WikiQACodePackage/code/process_data.py".
    It performs basic preprocessing (like tokenize) and also loads
    the pretrained Word2Vec results from Google.

    The command to run the script is:
    $ python -u process_data.py --w2v_fname ../data/GoogleNews-vectors-negative300.bin \
    --extract_feat 1 ../data/wiki/WikiQASent-train.txt ../data/wiki/WikiQASent-dev.txt \
    ../data/wiki/WikiQASent-test.txt ../wiki_cnn.pkl
    """
    revs, wordvecs, max_l = pickle.load(open('./data/wiki_cnn.pkl', 'rb'))

    # find_similar_words(wordvecs)
    # pdb.set_trace()

    # NOTE: keep this for being consistent with WikiQA code package
    # Truncate and pad all questions or sentences to this length.
    max_l = 40

    dataset = make_data(revs, wordvecs.word_idx_map, max_l=max_l)
    train_data, val_data, test_data = dataset
    # visualize_distribution(train_data) # aid choose buckets

    # SAMPLEBAG
    train_tuple_b = cast_data_to_buckets(train_data, BUCKETS)
    # train_tuple_b = cast_data_to_buckets(train_data, BUCKETS_COLLAPSED,
    #                                      if_pad_token_only=True)
    valid_tuple_b = cast_data_to_buckets(val_data, BUCKETS)
    test_tuple_b = cast_data_to_buckets(test_data, BUCKETS)

    # Vocabulary
    assert('<pad>' not in wordvecs.word_idx_map)
    assert(0 not in wordvecs.word_idx_map.values())
    word2id = wordvecs.word_idx_map
    word2id['<pad>'] = 0

    # Embeddings
    word_embeddings = wordvecs.W
    assert(len(word_embeddings) == len(word2id))

    return train_tuple_b, valid_tuple_b, test_tuple_b, word2id, word_embeddings



if __name__ == '__main__':
    train_tuple_b, valid_tuple_b, test_tuple_b, word2id, word_embeddings = \
            prep_data()
    pdb.set_trace()
