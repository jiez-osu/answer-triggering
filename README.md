# answer-triggering
Our paper "An End-to-End Deep Framework for Answer Triggeringwith a Novel Group-Level Objective" is accpeted by EMNLP 2017.

## Dataset preparation
We use the [WikiQA](http://research.microsoft.com/en-US/downloads/4495da01-db8c-4041-a7f6-7984a4f6a905/default.aspx) data set.
Please see the paper for more details:
[WikiQA: A Challenge Dataset for Open-Domain Question Answering](https://www.microsoft.com/en-us/research/publication/wikiqa-a-challenge-dataset-for-open-domain-question-answering/)

Please download the original [WikiQA code package](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwie8NqY35vVAhXEs1QKHcmJABQQFggqMAA&url=https%3A%2F%2Fwww.microsoft.com%2Fen-us%2Fdownload%2Fdetails.aspx%3Fid%3D52355&usg=AFQjCNEPkmGIkodGD8H9PV2ZpQb0NGz1mw), and run the following commands for preprocessing:
~~~
cd WikiQACodePackage/code
python -u process_data.py --w2v_fname ../data/GoogleNews-vectors-negative300.bin --extract_feat 1 ../data/wiki/WikiQASent-trai    n.txt ../data/wiki/WikiQASent-dev.txt ../data/wiki/WikiQASent-test.txt ../wiki_cnn.pkl
~~~
Our model use the exact same preprocessed data here: "../wiki_cnn.pkl" for fair comparison of performances.

## Running our model
We use tensorflow (0.12.1) to implement our NN model. 
Copy "../wiki_cnn.pkl" into "./data" before running the following command:
~~~
python run.py --train --plus_cnt=True
~~~
