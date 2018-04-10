from gensim.models import KeyedVectors

w2v_model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin',
                                              binary=True)




import numpy as np

w2v_vocab = len(w2v_model.vocab)
w2v_dim = 300




window_size = 3
phrase = 2 * window_size + 1

batch_size = 32




from keras.models import load_model

def load_model_from_file(file_name):
    model = load_model('models/{}.h5'.format(file_name))
    return model
    
def evaluate_model(model, X, Y):
    score = model.evaluate(X,
                           Y,
                           batch_size=batch_size)
    return score




from keras.utils.np_utils import to_categorical
from lxml import etree
from nltk.tokenize import sent_tokenize, word_tokenize
from pickle import load

train_or_test = 'test'
file = 'data/semeval2007/{0}/lexical-sample/english-lexical-sample.{0}.xml'.format(train_or_test)
root = etree.parse(file)

trained_lexelt_items = load('lexelts.txt')
instance_indexer = {} # { instance_id: X.index(instance_id) and Y.index(instance_id) }

for lexelt in root.findall('lexelt'):
    lexelt_item = lexelt.attrib['item']
    if lexelt_item not in trained_lexelt_items:
        continue
    lexelt_pos = lexelt.attrib['pos']
    instances = lexelt.findall('instance')

    number_of_instances = len(instances)
    X = np.zeros((number_of_instances, phrase, w2v_dim), dtype=np.float64)
    Y = np.zeros(number_of_instances, dtype=np.uint8)

    for instance_index, instance in enumerate(instances):
        instance_id = instance.attrib['id']
        instance_indexer[instance_id] = instance_index

        context = instance.find('context')
        head = context.find('head').text.strip()
        etree.strip_tags(context, 'head')
        words = list(map(lambda sentence: word_tokenize(sentence), sent_tokenize(context.text)))
        sentence_index, word_index = -1, -1
        for (s_index, sentence) in enumerate(words):
            for (w_index, word) in enumerate(sentence):
                if word == head:
                    sentence_index, word_index = s_index, w_index
                    break
        if sentence_index == -1 or word_index == -1: # Lexelt did not exist in the context
            continue

        sentence = words[sentence_index]
        lower_bound = max(0, word_index - window_size)
        upper_bound = min(word_index + window_size, len(sentence))
        w2v_vectors = np.empty((phrase, w2v_dim))
        for w_index in range(lower_bound, upper_bound):
            word = sentence[w_index]
            if word in w2v_model:
                w2v_vectors[w_index - lower_bound] = w2v_model[word] # Set to ref to this vector
        X[instance_index] = w2v_vectors

    model = load_model_from_file(lexelt_item)
    with open('data/semeval2007/key/english-lexical-sample.test.key') as key:
        for line in key:
            lexelt_item, instance_id, answer_sense_id = line.strip().split(' ')
            if instance_id not in instance_indexer:
                continue
            instance_index = instance_indexer[instance_id]
            Y[instance_index] = int(answer_sense_id)
    Y = to_categorical(Y)
    score = evaluate_model(model, X, Y)
    print(lexelt_item)
    print(score)
    print()
