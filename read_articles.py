import pandas as pd
import numpy as np
from articles import articles
from preprocessing import preprocess_text

##Note, this code was built with a codecademy course. There are modifications to pull the top n words instead of 1 which the cours did

# import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

# preprocess articles
processed_articles=[preprocess_text(story) for story in articles]

# initialize and fit CountVectorizer
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(processed_articles)


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


# create pandas DataFrame with word counts
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

try:
  df_tf_idf = pd.DataFrame(tfidf_scores.T.todense(), index=feature_names, columns=article_index)
  print(df_tf_idf)
except:
  pass

"""
filename = "DF-TF-IDF.csv"
f = open(filename, "w")

f.write(df_tf_idf + "\n")
"""

# get highest scoring tf-idf term for each article
for i in range(1,len(articles)+1):
  print(df_tf_idf[[f'Article {i}']].idxmax())
  n = 15
  print(df_tf_idf[[f'Article {i}']].nlargest(n,f'Article {i}'))
  
