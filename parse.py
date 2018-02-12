from lxml import etree
import nltk

from nltk.tokenize import sent_tokenize, word_tokenize

file = 'data/semeval2007/train/lexical-sample/english-lexical-sample.train.xml'
root = etree.parse(file)

for lexelt in root.findall('lexelt'):
    lexelt_item = lexelt.attrib['item'].split('.')[0]
    lexelt_pos = lexelt.attrib['pos']
    for instance in lexelt.findall('instance'):
        instance_id = instance.attrib['id']
        answer_sense_id = instance.find('answer').attrib['senseid']
        context = instance.find('context')
        etree.strip_tags(context, 'head')
        sentences = sent_tokenize(context.text)
        words = map(lambda sentence: word_tokenize(sentence), sentences)
        print(list(words))
        break
    break
