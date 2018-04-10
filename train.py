from gensim.models import KeyedVectors

w2v_model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin',
                                              binary=True)




import numpy as np

w2v_vocab = len(w2v_model.vocab)
w2v_dim = 300




window_size = 3
phrase = 2 * window_size + 1

batch_size = 32
epochs = 3




from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

input_length = phrase
input_shape = (phrase, w2v_dim)

'''
embedding_layer = Embedding(w2v_vocab,
                            w2v_dim,
                            input_length=input_length)
'''
dense_layer = Dense(1,
                    activation='softmax',
                    input_shape=input_shape)
layers = [
    LSTM(phrase,
         input_shape=input_shape,
         return_sequences=True),
    LSTM(phrase,
         input_shape=input_shape,
         return_sequences=True),
    LSTM(phrase,
         input_shape=input_shape),
    Dropout(0.5,
            seed=3000), # random seed
    dense_layer
]

def generate_model():
    model = Sequential()
    for layer in layers:
        model.add(layer)
    model.compile(optimizer='adam',
                  metrics=['accuracy'],
                  loss='categorical_crossentropy')
    return model

def fit_model(model, X, Y):
    model.fit(X,
              Y,
              batch_size=batch_size,
              epochs=epochs)
    return model

def save_model_to_file(model, file_name):
    model.save('models/{}.h5'.format(file_name))




from keras.utils.np_utils import to_categorical
from lxml import etree
from nltk.tokenize import sent_tokenize, word_tokenize
from pickle import dump

train_or_test = 'train'
file = 'data/semeval2007/{0}/lexical-sample/english-lexical-sample.{0}.xml'.format(train_or_test)
root = etree.parse(file)

trained_lexelt_items = set()
instance_indexer = {} # { instance_id: X.index(instance_id) and Y.index(instance_id) }

for lexelt in root.findall('lexelt'):
    lexelt_item = lexelt.attrib['item']
    lexelt_pos = lexelt.attrib['pos']
    instances = lexelt.findall('instance')

    number_of_instances = len(instances)
    X = np.zeros((number_of_instances, phrase, w2v_dim), dtype=np.float64)
    Y = np.zeros(number_of_instances, dtype=np.uint8)

    for instance_index, instance in enumerate(instances):
        instance_id = instance.attrib['id']
        instance_indexer[instance_id] = instance_index

        answer_sense_id = instance.find('answer').attrib['senseid']
        Y[instance_index] = int(answer_sense_id)

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

    Y = to_categorical(Y)
    number_of_classes = len(set(Y.flat))
    while True:
        try:
            dense_layer = Dense(number_of_classes, activation='softmax', input_shape=input_shape)
            layers[-1] = dense_layer
            model = generate_model()
            fit_model(model, X, Y)
            print(lexelt_item)
            print(model.summary())
            print()
            break
        except:
            number_of_classes += 1
    save_model_to_file(model, lexelt_item)
    trained_lexelt_items.add(lexelt_item)

dump(trained_lexelt_items, 'lexelts.txt')
