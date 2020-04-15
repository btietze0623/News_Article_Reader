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

This repository has 3 scripts. Read_articles.py, articles.py, and preprocessing.py. Articles contains 10 articles that I found online. Read_articles.py runs the tf-idf analysis and calls the preprocessing functions from preprocessing.py.

## Import libraries, scripts and preprocess data
```
import pandas as pd
import numpy as np
from articles import articles
from preprocessing import preprocess_text

##Note, this code was built with a codecademy course. There are modifications to pull the top n words instead of 1 which the cours did

#import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

#preprocess articles
processed_articles=[preprocess_text(story) for story in articles]

#initialize and fit CountVectorizer
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(processed_articles)
```


## Apply tf-idf vectorization to create scores

```
# convert counts to tf-idf
transformer = TfidfTransformer(norm=None)


# initialize and fit TfidfVectorizer
tfidf_scores_transformed = transformer.fit_transform(counts)

vectorizer = TfidfVectorizer(norm=None)
tfidf_scores = vectorizer.fit_transform(processed_articles)
# check if tf-idf scores are equal
if np.allclose(tfidf_scores_transformed.todense(), tfidf_scores.todense()):
  print(pd.DataFrame({'Are the tf-idf scores the same?':['YES']}))
else:
  print(pd.DataFrame({'Are the tf-idf scores the same?':['No, something is wrong :(']}))




# get vocabulary of terms
try:
  feature_names = vectorizer.get_feature_names()
except:
  pass

# get article index
try:
  article_index = [f"Article {i+1}" for i in range(len(articles))]
except:
  pass
```

## Create dataframes (one for term frequency and another for tf-idf scores)

```

# create pandas DataFrame with word counts - this is term frequency
try:
  df_word_counts = pd.DataFrame(counts.T.todense(), index=feature_names, columns=article_index)
  print(df_word_counts)
except:
  pass

# create pandas DataFrame(s) with tf-idf scores
try:
  df_tf_idf = pd.DataFrame(tfidf_scores_transformed.T.todense(), index=feature_names, columns=article_index)
  print(df_tf_idf)
except:
  pass


```

## Print tf-idf output for the top 15 results

```
#get highest scoring tf-idf term for each article
for i in range(1,len(articles)+1):
  print(df_tf_idf[[f'Article {i}']].idxmax())
  n = 15
  print(df_tf_idf[[f'Article {i}']].nlargest(n,f'Article {i}'))
```
