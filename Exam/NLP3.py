import nltk
nltk.download('punkt')
nltk.download('averaged_preceptron_tagger')
nltk.download('brown')

from nltk.tokenize import word_tokenize
sentence = "The brown fox jump over the lazy dog"
tokens = word_tokenize(sentence)
pos_tagg = nltk.pos_tag(tokens)
print("POS_TAGGERS:",pos_tagg)

from nltk.corpus import brown
from nltk.util import ngrams
text = brown.words(categories="news")[:1000]
bigrams = list(ngrams(text,2))
fre_dist = nltk.FreqDist(bigrams)

for bigram in bigrams:
    print("NGRAMS\n",f"{bigram} : {fre_dist[bigram]}")

from nltk.chunk import RegexpParser
tagger_sentences = nltk.pos_tag(word_tokenize("The brown fox jump over the lazy dog"))
grammer = r"NP: {<DT>?<JJ>*<NN>}"
cp = RegexpParser(grammer)
result = cp.parse(tagger_sentences)
print("CHUNKING\n",result)