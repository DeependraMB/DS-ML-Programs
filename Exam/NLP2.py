import nltk

nltk.download('punkt')
nltk.download('averaged_preceptron_tagger')
nltk.download('brown')

from nltk.tokenize import word_tokenize
sentence = "The brown fox jump over the lazy dog"
tokens = word_tokenize(sentence)
pos_tagg = nltk.pos_tag(tokens)
print("POS_TAGGER\n",pos_tagg)

from nltk.corpus import brown
from nltk.util import ngrams
text = brown.words(categories="news")[:1000]
bigrams = list(ngrams(text, 2))
freq_dis = nltk.FreqDist(bigrams)
for bigram in bigrams:
    print(f"{bigram}: {freq_dis[bigram]}")

from nltk.chunk import RegexpParser
tagged_sentences = nltk.pos_tag(word_tokenize("The brown fox jump over the lazy dog"))
grammer = r"NP: {<DT>?<JJ>*<NN>}"
cp = RegexpParser(grammer)
re=cp.parse(tagged_sentences)
print("Chunking",re)
