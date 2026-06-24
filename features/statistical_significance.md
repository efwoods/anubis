what if I wanted to say given 1000 generated texts there is no statistically significant difference and arbitrary generated and the corpus given a list of these features 95% of the time?

# Detailed Features yet to be used:
character n-grams
word n-grams
function word frequencies
punctuation frequencies
lexical richness (vocab size): unique_words vs. total_words
(FIND AND STORE KEY PHRASES) such as "you know" "got it." "what do ya mean?"
average sentence length (words_per_sentence)
average word length (characters per word)

# Best to implement open-set attribution: requires a rejection threshold to determine if zero or more authors wrote this document



# Background similar works

---

#### Mosteller and Wallace (inference and disputed authorship): Situation: Given anonymous documents in the Federalist Papers, what is the probablity that a document was written from a particular individual. bayesian probability; what is the probability a document d came from a corpus? 

Goal: P(corpus | D); use few function words instead of topics or content words. 

identify the P(corpus) (count frequencies of function words, aggregate, estimate probabilities (ratio of function words to total function words)); 

use same indicators on the document: count the function words, compute the likelihood using the sum of products of the log of probabilities of frequencies of the same function words to determine P(D | corpus_1 ). 

Finally: identify P(corpus | D) = P(D | corpus)*P(corpus) where P (corpus) is the estimation of the corpus and P(D | corpus) is the likelihood computation (observation of the function words; computed likelihood: sum of count per log of probabilities of a given frequency of a word for all words); 

# Koppel et all: list features to create a vector; use a model to train on this vector of features; use a threshold for rejection. 
