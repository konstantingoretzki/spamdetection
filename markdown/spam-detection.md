# Spam detection


```python
import pandas as pd
import numpy as np
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

# Preprocessing


```python
## get paths of all spam and ham files
# path change might be needed if run on linux

# get list of all spam files (spamassassin)
spam_path_spamassassin = ".\\datasets\\spamassassin\\spam"
spam_files_spamassassin = [join(spam_path_spamassassin, f) for f in listdir(spam_path_spamassassin) if isfile(join(spam_path_spamassassin, f))]

# get list of all valid files (spamassassin)
ham_path_spamassassin = ".\\datasets\\spamassassin\\ham"
ham_files_spamassassin = [join(ham_path_spamassassin, f) for f in listdir(ham_path_spamassassin) if isfile(join(ham_path_spamassassin, f))]

# get list of all spam files (trec07p)
spam_path_trec07p = ".\\datasets\\trec07p\\spam"
spam_files_trec07p = [join(spam_path_trec07p, f) for f in listdir(spam_path_trec07p) if isfile(join(spam_path_trec07p, f))]

# get list of all valid files (trec07p)
ham_path_trec07p = ".\\datasets\\trec07p\\ham"
ham_files_trec07p = [join(ham_path_trec07p, f) for f in listdir(ham_path_trec07p) if isfile(join(ham_path_trec07p, f))]
```


```python
# combine all spam files, combine all ham files
all_spam = spam_files_spamassassin + spam_files_trec07p
all_ham = ham_files_spamassassin + ham_files_trec07p
```


```python
def parse_email(email_raw):
    email_payload = email_raw.get_payload()
    
    email_body = ""

    if isinstance(email_payload, list):
        for part in email_payload:
            email_body += str(parse_email(part))

        return email_body
    else:
        if "Content-Type" in email_raw:
            if "text/html" in email_raw["Content-Type"].lower() or "text/plain" in email_raw["Content-Type"].lower(): # only parse content of type "text/html" and "text/plain"
                if "Content-Transfer-Encoding" in email_raw:
                    if email_raw["Content-Transfer-Encoding"].lower() == "base64": # check if its base64 encoded
                        try:
                            return str(base64.b64decode(email_payload))
                        except:       # if the decoding did not work
                            return "" # just return an empty string
                    elif email_raw["Content-Transfer-Encoding"].lower() == "quoted-printable":
                        try:
                            email_payload = ''.join(filter(lambda x: x in string.printable, email_payload))
                            return str(quopri.decodestring(email_payload))
                        except:       # if the decoding did not work
                            return "" # just return an empty string
                    else:
                        return email_payload
                else:
                    return email_payload
        elif email_raw.get_default_type() == "text/plain":
            # If the there is no "Content-Type" and the default type is "text/plain"
            return email_payload
        else:
            return ""
```


```python
def parse_dataset(dataset, is_spam):
    rows = []
    parser = email.parser.BytesParser()
    re_email = re.compile("[\w.-]+@[\w.-]+.[\w.-]+", re.UNICODE)

    for mail in dataset:
        with open(mail, "rb") as f:

            email_raw = parser.parse(f)

            subject_mail = email_raw['subject']
            from_mail = email_raw['From']
            if from_mail != None:
                try:
                    from_mail = re_email.search(str(from_mail)).group()
                except AttributeError:
                    print("Could not parse email 'from' header")
                    from_mail = None
            
            try:
                email_payload = parse_email(email_raw)
            except Exception as e:
                print("[-] Unknown email parse error")
                
            if email_payload == None:
                print("Could not parse email body")
            
            if email_payload == None or len(email_payload) == 0:
                email_payload = "0"

            urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[-$_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', email_payload)
            domains = []
            if len(urls) == 0:
                urls = 0
            else:
                for url in urls:
                    try:
                        domains.append(re.sub(":\d+", "", urllib.parse.urlparse(url).netloc))
                    except:
                        print("Could not parse domain") 
                
            if len(domains) == 0:
                domains = 0

            if BeautifulSoup(email_payload.encode("utf-8"), "html.parser").find():
                cleantext = BeautifulSoup(email_payload, "html.parser").text
            else:
                cleantext = email_payload
                
            clean_data = cleantext
            clean_data.replace("\n", " ")

            rows.append([is_spam, cleantext, subject_mail, from_mail, urls, domains])
            
    return rows
```


```python
model_spam_files = all_spam
model_ham_files = all_ham 
```


```python
rows_spam = parse_dataset(model_spam_files, 1)
rows_ham = parse_dataset(model_ham_files, 0)
```

    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email body
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email body
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email body
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email body
    Could not parse email body
    Could not parse email 'from' header
    Could not parse email body
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email body
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email body
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email body
    [-] Unknown email parse error
    [-] Unknown email parse error
    Could not parse email body
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email body
    Could not parse email body
    Could not parse email 'from' header
    Could not parse email body
    Could not parse email body
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email 'from' header
    Could not parse email body
    [-] Unknown email parse error
    Could not parse email body
    Could not parse email body
    Could not parse email body
    Could not parse email body
    Could not parse email body
    Could not parse email body
    Could not parse email body
    Wall time: 6min 13s
    


```python
# create dataframe
rows_all = rows_spam + rows_ham
df = pd.DataFrame(rows_all, columns = ['spam', 'raw_data', 'subject', 'from', 'urls', 'domains'])
```


```python
# reshuffle dataframes (currently spam and ham are strictly separated)
df = sklearn.utils.shuffle(df)
# fix indices
df = df.reset_index(drop=True)
```


```python
# add missing punkt
#nltk.download('punkt')

# add columns for amount of chars / words
df["chars"] = df["raw_data"].apply(len)
```


```python
def clean_text(text):
    # remove replies and forwards
    start_reply = re.search(r"\nOn .* wrote:", text)
    if start_reply != None:
        cleared_text = text[:start_reply.start()]
    else:
        cleared_text = text
    
    # remove \n or \r or \\n or \\r
    cleared_text = cleared_text.replace('\n', ' ').replace('\r', ' ').replace('\\n', ' ').replace('\\r', ' ')
    
    # remove URLs
    cleared_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[-$_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', cleared_text)
    
    # remove email addrs
    re_email = re.compile("[\w.-]+@[\w.-]+.[\w.-]+", re.UNICODE)
    cleared_text = re_email.sub(' ', cleared_text)
    
    # replace non-alpha chars with space
    cleared_text = re.sub('[^a-zA-Z]', ' ', cleared_text)
    
    # convert everything to lowercase
    cleared_text = cleared_text.lower()
    
    cleared_text = cleared_text.split()
    cleared_text = ' '.join(cleared_text)
    
    return cleared_text
```


```python
# add new column for cleaned data (no urls, email addrs, only alpha, all lowercase)
df["data"] = df["raw_data"].apply(clean_text)
```


```python
df["token_text"] = df.apply(lambda row: nltk.word_tokenize(str(row["data"])), axis=1)
```


```python
df['tokens']  = df['token_text'].str.len()
```


```python
# nltk.download('stopwords')

def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    filtered_text = [word for word in text if word not in stop_words]
    return filtered_text
```


```python
df["stop_text"] = df["token_text"].apply(remove_stopwords)
```


```python
#nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def lemmatize_word(text):
    lemmas = [lemmatizer.lemmatize(word, pos ='v') for word in text]
    return lemmas
```


```python
df["clean_text"] = df["stop_text"].apply(lemmatize_word)
```


```python
# Remove duplicates
df = df.drop_duplicates(subset="raw_data", keep="first")

# Remove emails with less then five words
print(f"All emails without dups: {len(df)}")
print(f"Emails with less then five words: {len(df[df['clean_text'].map(lambda d: len(d)) < 5])}")
df = df.drop(df[df['clean_text'].map(lambda d: len(d)) < 5].index)
print(f"Emails with at least 5 words: {len(df)}")
```

# Vocabulary


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

# Model


```python
df_spam = df[df["spam"] == 1]
```


```python
df_ham = df[df["spam"] == 0]
```


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


