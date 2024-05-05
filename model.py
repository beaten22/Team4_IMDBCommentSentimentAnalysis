#https://blog.tensorflow.org/2020/12/making-bert-easier-with-preprocessing-models-from-tensorflow-hub.html
#imported bert encoder and preprocessor from link
import tensorflow_hub as hub
import tensorflow_text as tf_text
preprocess = hub.load('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/1')
encoder = hub.load('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3')
def get_sentences_embeding(sentences):
    preprocess(sentences)
get_sentences_embeding([
    "it is a good and excellentmovie",
    "this was a terrible movie"
    ])