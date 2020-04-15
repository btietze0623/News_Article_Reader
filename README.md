# News_Article_Reader


## Use Case
Let's say you want to get the most relevant words of a news article without actually reading the article. This script does that using a TF-IDF analysis. 
A tf-idf analysis 

TF(t) = (Number of instances term t in a single document) / (Total number of terms in the document)
-this measures the count of the term t in a document, similar to an ngram analysis

IDF(t) = log_e(Total number of documents / Number of documents with term t in it)
-this measures how important the term t is to the specific document.

The tf-idf score is the product of those equations.
TF-IDF(t) = TF(t) * IDF(t)

I created this code through a Codecademy course, and made my own modifications to improve the output. Primarily, I extended the results of the tf-idf test from 1 word to n = 15. I have used an existing preprocessing.py script that codecademy built. This script lemmatizes, tokenizes, normalizes the corpus. I wrote previous code that did this in the News-Scraper-Topic-Analyzer respository


