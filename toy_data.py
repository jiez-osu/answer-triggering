import numpy as np


def fake_data_with_buckets():
	"""Fake data set for both 3 and 6 bucket."""
	# Each bucket is indicated by three numbers:
	# (1) sentence_number, (2) question_length, (3) sentence_length
	buckets = [[2,3,4], [5,3,6]]
	buckets_collapsed = [[3,4], [3,6]]
	bucket_sizes = [6, 3]
	train_tuple_b = []

	# First bucket
	questions = np.array([[1,1,0],[1,2,1],[2,3,4],[1,3,0],[7,8,6],[8,4,2]])
	question_features = np.array([[2],[3],[3],[2],[3],[3]])
	sentences = np.array([[[8,5,1,5],[8,2,1,5]],[[9,3,4,2],[7,4,1,2]],
												[[9,5,3,0],[7,5,6,0]],[[7,1,8,0],[9,3,4,0]],
												[[9,3,7,1],[0,0,0,0]],[[1,2,3,0],[0,0,0,0]]])
	sentence_features = np.array([[[4,2,3],[4,1,2]],[[4,2,1],[4,2,2]],
																[[3,2,1],[3,1,2]],[[3,1,2],[3,1,1]],
																[[4,3,1],[0,0,0]],[[3,1,1],[0,0,0]]])
	sentence_labels = np.array([[1,1],[1,0],[0,0],[0,0],[1,0],[0,0]])
	train_tuple_b.append([questions, sentences, sentence_labels, 
												question_features, sentence_features])

	# Second bucket
	questions = np.array([[5,2,0],[3,7,2],[7,2,1]])
	question_features = np.array([[2],[3],[3]])
	sentences = np.array([[[1,1,1,1,2,1],[2,3,4,1,3,6],[7,8,6,4,5,6],
												 [4,5,6,5,6,7],[8,9,2,4,6,0]],
												[[6,3,6,3,5,7],[4,7,2,9,3,2],[6,7,2,3,1,0],
												 [5,3,4,6,2,1],[0,0,0,0,0,0]],
												[[1,1,1,3,2,0],[2,5,3,5,9,3],[4,6,2,4,6,8],
												 [4,8,2,7,3,5],[0,0,0,0,0,0]]])
	sentence_features = np.array([[[6,3,2],[6,4,3],[6,2,1],[6,4,2],[5,1,1]],
																[[6,2,1],[6,1,1],[5,1,2],[6,6,2],[0,0,0]],
																[[5,3,1],[6,5,2],[6,5,2],[6,3,2],[0,0,0]]])
	sentence_labels = np.array([[0,0,0,0,0],
															[0,1,0,0,0],
															[1,1,1,0,0]])
	train_tuple_b.append([questions, sentences, sentence_labels,
												question_features, sentence_features])

	train_tuple_b = np.array(train_tuple_b)

	# word2id
	word2id = {'0':0, '1':1, '2':2, '3':3, '4':4,
						 '5':5, '6':6, '7':7, '8':8, '9':9}

	word_embeddings = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
															[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
															[0,1,2,0,0,0,0,0,0,0,0,0,0,0,0,0],
															[0,1,2,3,0,0,0,0,0,0,0,0,0,0,0,0],
															[0,1,2,3,4,0,0,0,0,0,0,0,0,0,0,0],
															[0,1,2,3,4,5,0,0,0,0,0,0,0,0,0,0],
															[0,1,2,3,4,5,6,0,0,0,0,0,0,0,0,0],
															[0,1,2,3,4,5,6,7,0,0,0,0,0,0,0,0],
															[0,1,2,3,4,5,6,7,8,0,0,0,0,0,0,0],
															[0,1,2,3,4,5,6,7,8,9,0,0,0,0,0,0]])
	return buckets, buckets_collapsed, bucket_sizes, train_tuple_b, word2id, word_embeddings
