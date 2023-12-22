import nltk
nltk.download('punkt')
nltk.download('brown')
nltk.download('averaged_preceptron_tagger')

from nltk.tokenize import word_tokenize
from nltk.chunk import RegexpParser
from nltk.corpus import brown
from nltk.util import ngrams

sentence = "The brown dog jump ove the lazy fox"
tokens = word_tokenize(sentence)
pos_tagg = nltk.pos_tag(tokens)
print("POS_Tagger :\n",pos_tagg)

text = brown.words(categories="news")[:1000]
bigrams = list(ngrams(text,2))
freq_dist = nltk.FreqDist(bigrams)
print("N_Grams\n")
for bigram in bigrams:
    print(f"{bigram} : {freq_dist[bigram]}")

tagged_sentences = nltk.pos_tag(word_tokenize("The brown dog jump ove the lazy fox"))
grammer = r"NP: {<DT>?<JJ>*<NN>}"
cp = RegexpParser(grammer)
result = cp.parse(tagged_sentences)
print("Chunking")
print(result)

