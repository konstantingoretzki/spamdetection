# Vocabulary


```python
import pandas as pd
from os import listdir
from os.path import isfile, join

import json
import re
from bs4 import BeautifulSoup
import email
import urllib
import base64
import string
import quopri

import seaborn as sns
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
# load clean dataframe
df = joblib.load("exports/dataframe.sav")
```

## Finding the best vocabulary


```python
# We want to have the biggest vocabulary (but only valid words) possible.
# This is needed because after training our model the vocabulary is fixed and can not be changed anymore.
```


```python
# All words of a huge wordlist can not be used because this would take way too long and take too many ressources (huge X matrix).
# The idea / tradeoff is to use all words from our known emails from the given datasets and remove all invalid ones.
```


```python
corpus = []
vocabulary = set()
for text in df["clean_text"]:
    msg = ' '.join([row for row in text])
    corpus.append(msg)
    for word in text:
        vocabulary.add(word)
```


```python
len(vocabulary)
```




    221767




```python
list(vocabulary)[:10]
```




    ['nlri',
     'ijyyjsigawq',
     'rucm',
     'tstudent',
     'bajativos',
     'schitannye',
     'drozhi',
     'liegrow',
     'resalable',
     'vynhnbucqlr']




```python
# there are strings in our vocabulary that are not valid (english) words
```


```python
# try to remove only unvalid words (keep as many valid words as possible)
```


```python
# idea: try different wordlists (and combine them) and make a intersection with the found words of the emails
# --> as much valid words as possible
```


```python
wordlist_nltk = words.words()
intersection_nltk = vocabulary.intersection(wordlist_nltk)
len(intersection_nltk)
```




    29142




```python
# with the nltk lib wordlist we get 29142 valid words 
```


```python
nltk.download('brown')
from nltk.corpus import brown
wordlist_brown = brown.words()
intersection_brown = vocabulary.intersection(wordlist_brown)
len(intersection_brown)
```

    [nltk_data] Downloading package brown to
    [nltk_data]     C:\Users\Konstantin\AppData\Roaming\nltk_data...
    [nltk_data]   Package brown is already up-to-date!
    




    18111




```python
# the brown wordlist removes more words, probably because this wordlist is pretty old (contains words that are not used anymore / in emails)
```


```python
wordlist_github = [] # https://github.com/dwyl/english-words
with open("resources/words-dwyl-github.txt", "r") as f:
    for line in f.readlines():
        wordlist_github.append(line[:-1])

intersection_github = vocabulary.intersection(wordlist_github)
len(intersection_github)
```




    36945




```python
# the github wordlist keeps more words
```


```python
wordlist_github_nltk_combined = wordlist_github + wordlist_nltk
intersection_github_nltk_combined = vocabulary.intersection(wordlist_github_nltk_combined)
len(intersection_github_nltk_combined)
```




    41012




```python
# combining the nltk and the github wordlist seems to be the best approach
```


```python
wordlist_github_nltk_combined_lower = [x.lower() for x in wordlist_github_nltk_combined]
intersection_github_nltk_combined_lower = vocabulary.intersection(wordlist_github_nltk_combined_lower)
len(intersection_github_nltk_combined_lower)
```




    55305




```python
# we get more words because we've lowered the choosen wordlist
# this is needed because our preprocessed text is only lowercase
# (words like "Example" in the wordlist will not be in the intersection because the vocabulary only contains lowercase words like "example")
```


```python
uppercase_words = 0
for x in wordlist_github_nltk_combined:
    if x[0] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        #print(x)
        uppercase_words += 1
print(uppercase_words)
```

    104119
    


```python
uppercase_words = 0
for x in wordlist_github_nltk_combined_lower:
    if x[0] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        #print(x)
        uppercase_words += 1
print(uppercase_words)
```

    0
    


```python
sym_diff_github_nltk_combined = sorted(intersection_github_nltk_combined_lower.symmetric_difference(intersection_github_nltk_combined))
```


```python
# how many uppercase words have been added due to lowering the words of the set
len(sym_diff_github_nltk_combined)
```




    14293


