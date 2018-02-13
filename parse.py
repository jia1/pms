from gensim.models import KeyedVectors

w2v_model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)




from lxml import etree

file = 'data/semeval2007/train/lexical-sample/english-lexical-sample.train.xml'
root = etree.parse(file)




from nltk.tokenize import sent_tokenize, word_tokenize

window_size = 5

for lexelt in root.findall('lexelt'):
    lexelt_item = lexelt.attrib['item'].split('.')[0]
    lexelt_pos = lexelt.attrib['pos']
    for instance in lexelt.findall('instance'):
        instance_id = instance.attrib['id']
        answer_sense_id = instance.find('answer').attrib['senseid']
        context = instance.find('context')
        head = context.find('head').text.strip()
        etree.strip_tags(context, 'head')
        sentences = sent_tokenize(context.text)
        words = list(map(lambda sentence: word_tokenize(sentence), sentences))
        sentence_index, word_index = -1, -1
        for (s_index, sentence) in enumerate(words):
            for (w_index, word) in enumerate(sentence):
                if word == head:
                    sentence_index, word_index = s_index, w_index
                    break
        if sentence_index == -1 or word_index == -1:
            continue
        sentence = words[sentence_index]
        left_bound = max(0, word_index - window_size)
        right_bound = min(word_index + window_size, len(sentence))
        left_pad = 0
        w2v_vectors = []
        for w_index in range(left_bound, right_bound):
            word = sentence[w_index]
            print(word)
            if word in w2v_model:
                vector = w2v_model[word]
                w2v_vectors.append(vector)
            else:
                if w_index < left_bound + window_size:
                    left_pad += 1
        right_pad = 2 * window_size + 1 - left_pad - len(w2v_vectors)
        input_vectors = [None for i in range(left_pad)] + w2v_vectors + [None for i in range(right_pad)]
        print(input_vectors)
        break
    break
