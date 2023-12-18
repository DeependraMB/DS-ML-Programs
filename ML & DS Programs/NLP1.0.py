import nltk
nltk.download('brown')
nltk.download('punkt' , download_dir="\nltk-3.8.1\nltk-3.8.1")
nltk.download('averaged_perceptron_tagger')

from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.corpus import brown
from nltk.chunk import RegexpParser

# Part-of-Speech Tagging
sentence = "The quick brown fox jumps over the lazy dog"
tokens = word_tokenize(sentence)
print("Part-of-Speech Tagging:")
pos_tags = nltk.pos_tag(tokens)
print(pos_tags)

# N-gram Analysis (Bigrams with Smoothing)
text = brown.words(categories='news')[:1000]
bigrams = list(ngrams(text, 2))
freq_dist = nltk.FreqDist(bigrams)
print("\nN-gram Analysis (Bigrams with Smoothing):")
for bigram in bigrams:
    print(f"{bigram}: {freq_dist[bigram]}")

# Chunking with Regular Expressions and POS tags
tagged_sentence = nltk.pos_tag(word_tokenize("The quick brown fox jumps over the lazy dog"))
grammar = r"NP: {<DT>?<JJ>*<NN>}"
cp = RegexpParser(grammar)
result = cp.parse(tagged_sentence)
print("\nChunking with Regular Expressions and POS tags:")
print(result)
