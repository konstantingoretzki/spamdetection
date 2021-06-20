# Model


```python
import pandas as pd

import numpy as np
import sklearn.utils

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import words

from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline    
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, plot_confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn import metrics
```


```python
# import dataframe
df = joblib.load("exports/dataframe.sav")
```

# Build vocabulary
Build the vocabulary over all words of our dataset, because the vocabulary can not be changed after the model is trained.


```python
collected_words = set()
for text in df["clean_text"]:
    for word in text:
        collected_words.add(word)

wordlist_nltk = words.words()

wordlist_github = [] # https://github.com/dwyl/english-words
with open("resources/words-dwyl-github.txt", "r") as f:
    for line in f.readlines():
        wordlist_github.append(line[:-1])

wordlist_github_nltk_combined = wordlist_github + wordlist_nltk

wordlist_github_nltk_combined_lower = [x.lower() for x in wordlist_github_nltk_combined]
intersection_github_nltk_combined_lower = collected_words.intersection(wordlist_github_nltk_combined_lower)
vocabulary = intersection_github_nltk_combined_lower
len(vocabulary)

joblib.dump(vocabulary, "exports/vocab.sav")
```




    ['exports/vocab.sav']



# Prepare dataset parts


```python
len(df)
```




    62701




```python
df_spam = df[df["spam"] == 1]
```


```python
df_ham = df[df["spam"] == 0]
```


```python
len(df_spam)
```




    34291




```python
len(df_ham)
```




    28410




```python
# use 25k for balanced dataset
amount = 25000
```


```python
# reshuffle dataframes (currently spam and ham are strictly separated)
df_spam = sklearn.utils.shuffle(df_spam)
df_ham = sklearn.utils.shuffle(df_ham)

# fix indices
df_spam = df_spam.reset_index(drop=True)
df_ham = df_ham.reset_index(drop=True)
```


```python
# create balanced df (25k spam and 25k ham mails) for training the model
df_model_train = df_spam[:amount].append(df_ham[:amount])
df_model_train = sklearn.utils.shuffle(df_model_train)
df_model_train = df_model_train.reset_index(drop=True)
```


```python
# use the rest of df_spam and df_ham but keep the new df (df_model_test) balanced
df_model_test = df_spam[amount:28410].append(df_ham[amount:28410])
df_model_test = sklearn.utils.shuffle(df_model_test)
df_model_test = df_model_test.reset_index(drop=True)
```


```python
# used for cross-validation, contains train and test dataset, balanced
df_balanced = df_model_train.append(df_model_test)
df_balanced = sklearn.utils.shuffle(df_balanced)
df_balanced = df_balanced.reset_index(drop=True)
```

# Estimator test


```python
corpus = []
for text in df_model_train["clean_text"]:
    msg = ' '.join([row for row in text])
    corpus.append(msg)
```


```python
tfidf = TfidfVectorizer(vocabulary=vocabulary)
x_train = tfidf.fit_transform(corpus).toarray()
y_train = df_model_train["spam"]
```


```python
corpus = []
for text in df_model_test["clean_text"]:
    msg = ' '.join([row for row in text])
    corpus.append(msg)
```


```python
x_test = tfidf.fit_transform(corpus).toarray()
y_test = df_model_test["spam"]
```


```python
import time
# try different estimators for the model 
classifiers = [MultinomialNB(), 
               RandomForestClassifier(),
               KNeighborsClassifier(),
               #SVC()
              ]

for cls in classifiers:
    time_start = time.time()
    cls.fit(x_train, y_train)
    time_end = time.time()
    print("Estimator:", cls)
    print("Train-Duration:", time_end - time_start, "s")
    score = cls.score(x_test, y_test)
    print("Accuracy:", score, "\n")
```

    Estimator: MultinomialNB()
    Train-Duration: 8.211855173110962 s
    Accuracy: 0.9523460410557185 
    
    Estimator: RandomForestClassifier()
    Train-Duration: 555.2905266284943 s
    Accuracy: 0.978592375366569 
    
    Estimator: KNeighborsClassifier()
    Train-Duration: 8.466284036636353 s
    Accuracy: 0.7541055718475074 
    
    


```python
# SVC() will no longer be considered due to extremly long run time
# after 8 hours it still hasn't finished ...
#classifiers = classifiers[:-1] # remove SVC() from list (if not uncomment in classifiers)
```


```python
# RandomForest worked the best (highest accuracy) and was the most efficient one for our tests
```

# Cross-validation


```python
corpus = []
for text in df_balanced["clean_text"]:
    msg = ' '.join([row for row in text])
    corpus.append(msg)
```


```python
tfidf = TfidfVectorizer(vocabulary=vocabulary)
x_all = tfidf.fit_transform(corpus).toarray()
y_all = df_balanced["spam"]
```


```python
x_all
```




    array([[0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.]])




```python
import time
# cross-validation using only clean_text with TFIDF as features
for cls in classifiers:
    time_start = time.time()
    cv_score = cross_val_score(cls, x_all,y_all,scoring="accuracy", cv=10)
    time_end = time.time()
    print("Estimator:", cls)
    print("Crossval-Duration:", time_end - time_start, "s")
    mean_score = cv_score.mean()
    print("Mean accuracy:", mean_score, "\n")
```

    Estimator: MultinomialNB()
    Crossval-Duration: 913.7169682979584 s
    Mean accuracy: 0.950703977472721 
    
    Estimator: RandomForestClassifier()
    Crossval-Duration: 5695.442259311676 s
    Mean accuracy: 0.9771383315733896 
    
    Estimator: KNeighborsClassifier()
    Crossval-Duration: 2097.342945575714 s
    Mean accuracy: 0.7539246744104189 
    
    


```python
# so looks like RandomForest performs best
# lets try to add more features and do the cross-validation again to see if this improves the accuracy
```


```python
x_all = np.append(x_all, df_balanced[["tokens", "chars"]].to_numpy(), axis=1)
```


```python
x_all
```




    array([[   0.,    0.,    0., ...,    0.,   77.,  489.],
           [   0.,    0.,    0., ...,    0.,  161., 1528.],
           [   0.,    0.,    0., ...,    0.,   61.,  433.],
           ...,
           [   0.,    0.,    0., ...,    0.,  157., 1036.],
           [   0.,    0.,    0., ...,    0.,  238., 1682.],
           [   0.,    0.,    0., ...,    0.,  196., 1159.]])




```python
# cross-validation using clean_text with TFIDF AND number of tokens and number of chars as features
for cls in classifiers:
    time_start = time.time()
    cv_score = cross_val_score(cls, x_all,y_all,scoring="accuracy", cv=10)
    time_end = time.time()
    print("Estimator:", cls)
    print("Crossval-Duration:", time_end - time_start, "s")
    mean_score = cv_score.mean()
    print("Mean accuracy:", mean_score, "\n")
```

    Estimator: MultinomialNB()
    Crossval-Duration: 1038.6484024524689 s
    Mean accuracy: 0.8724744808166138 
    
    Estimator: RandomForestClassifier()
    Crossval-Duration: 5761.310852527618 s
    Mean accuracy: 0.9793206617388244 
    
    Estimator: KNeighborsClassifier()
    Crossval-Duration: 2052.179085254669 s
    Mean accuracy: 0.7200281590989088 
    
    


```python
# RandomForest is still the best
```


```python
# by using more features (#tokens, #chars) we could improve the accuracy
```


```python
# RandomForest only clean_text: 0.9771383315733896
```


```python
# RandomForest clean_text, #chars, #tokens: 0.9793206617388244
```

# Export final model


```python
# ours experiments showed that the best estimator is RandomForest with TFIDF AND #tokens, #chars
```


```python
corpus = []
for text in df_model_train["clean_text"]:
    msg = ' '.join([row for row in text])
    corpus.append(msg)
```


```python
tfidf = TfidfVectorizer(vocabulary=vocabulary)
x_train = tfidf.fit_transform(corpus).toarray()
x_train = np.append(x_train, df_model_train[["tokens", "chars"]].to_numpy(), axis=1)
y_train = df_model_train["spam"]
```


```python
corpus = []
for text in df_model_test["clean_text"]:
    msg = ' '.join([row for row in text])
    corpus.append(msg)
```


```python
x_test = tfidf.fit_transform(corpus).toarray()
x_test = np.append(x_test, df_model_test[["tokens", "chars"]].to_numpy(), axis=1)
y_test = df_model_test["spam"]
```


```python
import time
classifiers = [RandomForestClassifier()]

for cls in classifiers:
    time_start = time.time()
    cls.fit(x_train, y_train)
    time_end = time.time()
    print("Estimator:", cls)
    print("Train-Duration:", time_end - time_start, "s")
    score = cls.score(x_test, y_test)
    print("Accuracy:", score, "\n")
```

    Estimator: RandomForestClassifier()
    Train-Duration: 498.00077414512634 s
    Accuracy: 0.9787390029325513 
    
    


```python
# export model
joblib.dump(classifiers[0], "exports/model.sav")
```




    ['exports/model.sav']


