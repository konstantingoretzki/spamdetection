# Preprocessing


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
# check lengths of files
len(spam_files_spamassassin)
len(ham_files_spamassassin)
len(spam_files_trec07p)
len(ham_files_trec07p)
```




    25220




```python
# combine all spam files, combine all ham files
all_spam = spam_files_spamassassin + spam_files_trec07p
all_ham = ham_files_spamassassin + ham_files_trec07p
```


```python
all_spam[-50:]
```




    ['.\\datasets\\trec07p\\spam\\inmail.9939',
     '.\\datasets\\trec07p\\spam\\inmail.994',
     '.\\datasets\\trec07p\\spam\\inmail.9940',
     '.\\datasets\\trec07p\\spam\\inmail.9941',
     '.\\datasets\\trec07p\\spam\\inmail.9942',
     '.\\datasets\\trec07p\\spam\\inmail.9943',
     '.\\datasets\\trec07p\\spam\\inmail.9944',
     '.\\datasets\\trec07p\\spam\\inmail.9945',
     '.\\datasets\\trec07p\\spam\\inmail.9946',
     '.\\datasets\\trec07p\\spam\\inmail.9947',
     '.\\datasets\\trec07p\\spam\\inmail.9948',
     '.\\datasets\\trec07p\\spam\\inmail.9949',
     '.\\datasets\\trec07p\\spam\\inmail.995',
     '.\\datasets\\trec07p\\spam\\inmail.9950',
     '.\\datasets\\trec07p\\spam\\inmail.9951',
     '.\\datasets\\trec07p\\spam\\inmail.9952',
     '.\\datasets\\trec07p\\spam\\inmail.9954',
     '.\\datasets\\trec07p\\spam\\inmail.9956',
     '.\\datasets\\trec07p\\spam\\inmail.9957',
     '.\\datasets\\trec07p\\spam\\inmail.996',
     '.\\datasets\\trec07p\\spam\\inmail.9961',
     '.\\datasets\\trec07p\\spam\\inmail.9962',
     '.\\datasets\\trec07p\\spam\\inmail.9963',
     '.\\datasets\\trec07p\\spam\\inmail.9964',
     '.\\datasets\\trec07p\\spam\\inmail.9966',
     '.\\datasets\\trec07p\\spam\\inmail.9968',
     '.\\datasets\\trec07p\\spam\\inmail.9969',
     '.\\datasets\\trec07p\\spam\\inmail.997',
     '.\\datasets\\trec07p\\spam\\inmail.9970',
     '.\\datasets\\trec07p\\spam\\inmail.9972',
     '.\\datasets\\trec07p\\spam\\inmail.9973',
     '.\\datasets\\trec07p\\spam\\inmail.9974',
     '.\\datasets\\trec07p\\spam\\inmail.9975',
     '.\\datasets\\trec07p\\spam\\inmail.9977',
     '.\\datasets\\trec07p\\spam\\inmail.9979',
     '.\\datasets\\trec07p\\spam\\inmail.998',
     '.\\datasets\\trec07p\\spam\\inmail.9980',
     '.\\datasets\\trec07p\\spam\\inmail.9981',
     '.\\datasets\\trec07p\\spam\\inmail.9982',
     '.\\datasets\\trec07p\\spam\\inmail.9983',
     '.\\datasets\\trec07p\\spam\\inmail.9984',
     '.\\datasets\\trec07p\\spam\\inmail.9985',
     '.\\datasets\\trec07p\\spam\\inmail.9989',
     '.\\datasets\\trec07p\\spam\\inmail.9991',
     '.\\datasets\\trec07p\\spam\\inmail.9993',
     '.\\datasets\\trec07p\\spam\\inmail.9995',
     '.\\datasets\\trec07p\\spam\\inmail.9996',
     '.\\datasets\\trec07p\\spam\\inmail.9997',
     '.\\datasets\\trec07p\\spam\\inmail.9998',
     '.\\datasets\\trec07p\\spam\\inmail.9999']




```python
print("Spam Mails:", len(all_spam))
print("Ham Mails:", len(all_ham))
```

    Spam Mails: 52596
    Ham Mails: 32171
    


```python
ywerte = [len(all_ham), len(all_spam)]
xwerte = [0, 1]
plt.bar(xwerte, ywerte)
plt.xlabel("Ham, Spam")
plt.ylabel("Amount")
plt.show()
```


    
![png](output_7_0.png)
    



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
%%time
rows_spam = parse_dataset(model_spam_files, 1)
#rows_spam[2]
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
%%time
rows_ham = parse_dataset(model_ham_files, 0)
#rows_ham[2]
```

    c:\users\konstantin\appdata\local\programs\python\python39\lib\site-packages\bs4\__init__.py:417: MarkupResemblesLocatorWarning: "http://www.post-gazette.com/columnists/20020905brian5.asp
    
    
    
    " looks like a URL. Beautiful Soup is not an HTTP client. You should probably use an HTTP client like requests to get the document behind the URL, and feed that document to Beautiful Soup.
      warnings.warn(
    c:\users\konstantin\appdata\local\programs\python\python39\lib\site-packages\bs4\__init__.py:417: MarkupResemblesLocatorWarning: "http://yenibiris.sendeyolla.com/medyadetay.aspx?&tid=3&cid=57&id=61365
    
    
    
    " looks like a URL. Beautiful Soup is not an HTTP client. You should probably use an HTTP client like requests to get the document behind the URL, and feed that document to Beautiful Soup.
      warnings.warn(
    

    Could not parse email 'from' header
    Wall time: 4min 28s
    


```python
# create dataframe
rows_all = rows_spam + rows_ham
df = pd.DataFrame(rows_all, columns = ['spam', 'raw_data', 'subject', 'from', 'urls', 'domains'])
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>spam</th>
      <th>raw_data</th>
      <th>subject</th>
      <th>from</th>
      <th>urls</th>
      <th>domains</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Greetings!\n\nYou are receiving this letter be...</td>
      <td>[ILUG] STOP THE MLM INSANITY</td>
      <td>startnow2002@hotmail.com</td>
      <td>[http://www.linux.ie]</td>
      <td>[www.linux.ie]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>b'\n\n\n\n\n\n  \n  \n    \n      \n    \n  \n...</td>
      <td>Life Insurance - Why Pay More?</td>
      <td>12a1mailbot1@web.de</td>
      <td>[http://website.e365.cc]</td>
      <td>[website.e365.cc]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>\n\n\n\n\n\nThe Need For Safety Is Real In 200...</td>
      <td>Real Protection, Stun Guns!  Free Shipping! Ti...</td>
      <td>lmrn@mailexcite.com</td>
      <td>[http://www.geocities.com, http://www.geocitie...</td>
      <td>[www.geocities.com, www.geocities.com, www.geo...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1) Fight The Risk of Cancer!\nhttp://www.adcli...</td>
      <td>[ILUG] Guaranteed to lose 10-12 lbs in 30 days...</td>
      <td>taylor@s3.serveimage.com</td>
      <td>[http://www.adclick.ws, http://www.adclick.ws,...</td>
      <td>[www.adclick.ws, www.adclick.ws, www.adclick.w...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1) Fight The Risk of Cancer!\nhttp://www.adcli...</td>
      <td>Guaranteed to lose 10-12 lbs in 30 days       ...</td>
      <td>sabrina@mx3.1premio.com</td>
      <td>[http://www.adclick.ws, http://www.adclick.ws,...</td>
      <td>[www.adclick.ws, www.adclick.ws, www.adclick.w...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>84762</th>
      <td>0</td>
      <td>Author: metze\nDate: 2007-04-16 07:41:12 +0000...</td>
      <td>svn commit: samba r22248 - in\n\tbranches/SAMB...</td>
      <td>metze@samba.org</td>
      <td>[http://websvn.samba.org]</td>
      <td>[websvn.samba.org]</td>
    </tr>
    <tr>
      <th>84763</th>
      <td>0</td>
      <td>On 4/9/07, Tom Phoenix  wrote:\n&gt;\n&gt; On 4/9/07...</td>
      <td>Re: OT: html checkbox question</td>
      <td>jm5379@gmail.com</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>84764</th>
      <td>0</td>
      <td>\n\nCharlie wrote:\n&gt; Hi, this is Charlie and ...</td>
      <td>Re: [R] Question for install Rapache package.</td>
      <td>ligges@statistik.uni-dortmund.de</td>
      <td>[https://stat.ethz.ch, http://www.R-project.or...</td>
      <td>[stat.ethz.ch, www.R-project.org, stat.ethz.ch...</td>
    </tr>
    <tr>
      <th>84765</th>
      <td>0</td>
      <td>Author: kseeger\nDate: 2007-04-16 07:47:27 +00...</td>
      <td>svn commit: samba-docs r1098 - in trunk: manpa...</td>
      <td>kseeger@samba.org</td>
      <td>[http://websvn.samba.org, http://www.samba.org...</td>
      <td>[websvn.samba.org, www.samba.org, www.samba.org]</td>
    </tr>
    <tr>
      <th>84766</th>
      <td>0</td>
      <td>**********************************************...</td>
      <td>CBS SportsLine Daily Sports Report</td>
      <td>mailer@mailer-relay.sportsline.com</td>
      <td>[http://ww1.sportsline.com, http://ww1.sportsl...</td>
      <td>[ww1.sportsline.com, ww1.sportsline.com, ww1.s...</td>
    </tr>
  </tbody>
</table>
<p>84767 rows × 6 columns</p>
</div>




```python
# reshuffle dataframes (currently spam and ham are strictly separated)
df = sklearn.utils.shuffle(df)
# fix indices
df = df.reset_index(drop=True)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>spam</th>
      <th>raw_data</th>
      <th>subject</th>
      <th>from</th>
      <th>urls</th>
      <th>domains</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>On Tue, 17 Apr 2007, James W. MacDonald wrote:...</td>
      <td>Re: [R] Greek symbols in xtable rows</td>
      <td>Roger.Bivand@nhh.no</td>
      <td>[https://stat.ethz.ch, http://www.R-project.or...</td>
      <td>[stat.ethz.ch, www.R-project.org, stat.ethz.ch...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Mon, May 14, 2...</td>
      <td>rent, Rent-A-List Special Reminder.</td>
      <td>noloop@rent-a-list.com</td>
      <td>[http://imttrack.com, http://imttrack.com, htt...</td>
      <td>[imttrack.com, imttrack.com, imttrack.com, www...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Does Size Matter7\n----\n\n60% of WOMEN said t...</td>
      <td>Is bunola in geneautry</td>
      <td>ssjhoneyville@mpinet.net</td>
      <td>[http://www.feruz.hk]</td>
      <td>[www.feruz.hk]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>IRRESISTABLE FORMULA TO \nGET RICHAND STAY RIC...</td>
      <td>Seven Figure Income Seeker</td>
      <td>adcos2@walla.com</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>If you use lme, you can fit a general correlat...</td>
      <td>Re: [R] ANOVA non-sphericity test and correcti...</td>
      <td>s.blomberg1@uq.edu.au</td>
      <td>[https://stat.ethz.ch, http://www.R-project.or...</td>
      <td>[stat.ethz.ch, www.R-project.org, stat.ethz.ch...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>49995</th>
      <td>0</td>
      <td>bitbitch@magnesium.net wrote:\n\n&gt;&gt;&gt;Wishful th...</td>
      <td>Re: A moment of silence for the First Amendmen...</td>
      <td>owen@permafrost.net</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>49996</th>
      <td>1</td>
      <td>Does Size Matter7\n___\n\n60% of WOMEN said th...</td>
      <td>He in adamsburg</td>
      <td>lmobjack@careernow.info</td>
      <td>[http://www.atgaros.com]</td>
      <td>[www.atgaros.com]</td>
    </tr>
    <tr>
      <th>49997</th>
      <td>1</td>
      <td>\n\n\n\nDear       sirs,\nAegis Capital       ...</td>
      <td>Job: Just For You.</td>
      <td>Saundra.Cook@computermail.net</td>
      <td>[http://joboffer-566115.aecapitall.hk, http://...</td>
      <td>[joboffer-566115.aecapitall.hk, joboffer-56611...</td>
    </tr>
    <tr>
      <th>49998</th>
      <td>1</td>
      <td>Does Size Matter?\n-----\n\n60% of WOMEN said ...</td>
      <td>My an elcho</td>
      <td>jevarts@geld-ins-haus.de</td>
      <td>[http://www.kliva.hk]</td>
      <td>[www.kliva.hk]</td>
    </tr>
    <tr>
      <th>49999</th>
      <td>0</td>
      <td>\nOn Apr 12, 2007, at 1:54 PM, Nicholas Clark ...</td>
      <td>Re: Limiting Exported Symbols on GCC</td>
      <td>jrisom@gmail.com</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>50000 rows × 6 columns</p>
</div>




```python
# get raw_data from column of index=3995
#df.iloc[3995].raw_data
```


```python
# show distribution of spam / ham emails
sns.countplot(df.spam)
```

    c:\users\konstantin\appdata\local\programs\python\python39\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    




    <AxesSubplot:xlabel='spam', ylabel='count'>




    
![png](output_17_2.png)
    



```python
# add missing punkt
#nltk.download('punkt')

# add columns for amount of chars / words
df["chars"] = df["raw_data"].apply(len)
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>spam</th>
      <th>raw_data</th>
      <th>subject</th>
      <th>from</th>
      <th>urls</th>
      <th>domains</th>
      <th>chars</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Greetings!\n\nYou are receiving this letter be...</td>
      <td>[ILUG] STOP THE MLM INSANITY</td>
      <td>startnow2002@hotmail.com</td>
      <td>[http://www.linux.ie]</td>
      <td>[www.linux.ie]</td>
      <td>3027</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>b'\n\n\n\n\n\n  \n  \n    \n      \n    \n  \n...</td>
      <td>Life Insurance - Why Pay More?</td>
      <td>12a1mailbot1@web.de</td>
      <td>[http://website.e365.cc]</td>
      <td>[website.e365.cc]</td>
      <td>1423</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>\n\n\n\n\n\nThe Need For Safety Is Real In 200...</td>
      <td>Real Protection, Stun Guns!  Free Shipping! Ti...</td>
      <td>lmrn@mailexcite.com</td>
      <td>[http://www.geocities.com, http://www.geocitie...</td>
      <td>[www.geocities.com, www.geocities.com, www.geo...</td>
      <td>4349</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1) Fight The Risk of Cancer!\nhttp://www.adcli...</td>
      <td>[ILUG] Guaranteed to lose 10-12 lbs in 30 days...</td>
      <td>taylor@s3.serveimage.com</td>
      <td>[http://www.adclick.ws, http://www.adclick.ws,...</td>
      <td>[www.adclick.ws, www.adclick.ws, www.adclick.w...</td>
      <td>781</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1) Fight The Risk of Cancer!\nhttp://www.adcli...</td>
      <td>Guaranteed to lose 10-12 lbs in 30 days       ...</td>
      <td>sabrina@mx3.1premio.com</td>
      <td>[http://www.adclick.ws, http://www.adclick.ws,...</td>
      <td>[www.adclick.ws, www.adclick.ws, www.adclick.w...</td>
      <td>636</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>84762</th>
      <td>0</td>
      <td>Author: metze\nDate: 2007-04-16 07:41:12 +0000...</td>
      <td>svn commit: samba r22248 - in\n\tbranches/SAMB...</td>
      <td>metze@samba.org</td>
      <td>[http://websvn.samba.org]</td>
      <td>[websvn.samba.org]</td>
      <td>840</td>
    </tr>
    <tr>
      <th>84763</th>
      <td>0</td>
      <td>On 4/9/07, Tom Phoenix  wrote:\n&gt;\n&gt; On 4/9/07...</td>
      <td>Re: OT: html checkbox question</td>
      <td>jm5379@gmail.com</td>
      <td>0</td>
      <td>0</td>
      <td>562</td>
    </tr>
    <tr>
      <th>84764</th>
      <td>0</td>
      <td>\n\nCharlie wrote:\n&gt; Hi, this is Charlie and ...</td>
      <td>Re: [R] Question for install Rapache package.</td>
      <td>ligges@statistik.uni-dortmund.de</td>
      <td>[https://stat.ethz.ch, http://www.R-project.or...</td>
      <td>[stat.ethz.ch, www.R-project.org, stat.ethz.ch...</td>
      <td>1178</td>
    </tr>
    <tr>
      <th>84765</th>
      <td>0</td>
      <td>Author: kseeger\nDate: 2007-04-16 07:47:27 +00...</td>
      <td>svn commit: samba-docs r1098 - in trunk: manpa...</td>
      <td>kseeger@samba.org</td>
      <td>[http://websvn.samba.org, http://www.samba.org...</td>
      <td>[websvn.samba.org, www.samba.org, www.samba.org]</td>
      <td>12536</td>
    </tr>
    <tr>
      <th>84766</th>
      <td>0</td>
      <td>**********************************************...</td>
      <td>CBS SportsLine Daily Sports Report</td>
      <td>mailer@mailer-relay.sportsline.com</td>
      <td>[http://ww1.sportsline.com, http://ww1.sportsl...</td>
      <td>[ww1.sportsline.com, ww1.sportsline.com, ww1.s...</td>
      <td>8518</td>
    </tr>
  </tbody>
</table>
<p>84767 rows × 7 columns</p>
</div>




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
%%time
# add new column for cleaned data (no urls, email addrs, only alpha, all lowercase)
df["data"] = df["raw_data"].apply(clean_text)
```

    Wall time: 12.1 s
    


```python
%%time
df["token_text"] = df.apply(lambda row: nltk.word_tokenize(str(row["data"])), axis=1)
```

    Wall time: 43.3 s
    


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>spam</th>
      <th>raw_data</th>
      <th>subject</th>
      <th>from</th>
      <th>urls</th>
      <th>domains</th>
      <th>chars</th>
      <th>data</th>
      <th>token_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Greetings!\n\nYou are receiving this letter be...</td>
      <td>[ILUG] STOP THE MLM INSANITY</td>
      <td>startnow2002@hotmail.com</td>
      <td>[http://www.linux.ie]</td>
      <td>[www.linux.ie]</td>
      <td>3027</td>
      <td>greetings you are receiving this letter becaus...</td>
      <td>[greetings, you, are, receiving, this, letter,...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>b'\n\n\n\n\n\n  \n  \n    \n      \n    \n  \n...</td>
      <td>Life Insurance - Why Pay More?</td>
      <td>12a1mailbot1@web.de</td>
      <td>[http://website.e365.cc]</td>
      <td>[website.e365.cc]</td>
      <td>1423</td>
      <td>b save up to on life insurance why spend more ...</td>
      <td>[b, save, up, to, on, life, insurance, why, sp...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>\n\n\n\n\n\nThe Need For Safety Is Real In 200...</td>
      <td>Real Protection, Stun Guns!  Free Shipping! Ti...</td>
      <td>lmrn@mailexcite.com</td>
      <td>[http://www.geocities.com, http://www.geocitie...</td>
      <td>[www.geocities.com, www.geocities.com, www.geo...</td>
      <td>4349</td>
      <td>the need for safety is real in you might only ...</td>
      <td>[the, need, for, safety, is, real, in, you, mi...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1) Fight The Risk of Cancer!\nhttp://www.adcli...</td>
      <td>[ILUG] Guaranteed to lose 10-12 lbs in 30 days...</td>
      <td>taylor@s3.serveimage.com</td>
      <td>[http://www.adclick.ws, http://www.adclick.ws,...</td>
      <td>[www.adclick.ws, www.adclick.ws, www.adclick.w...</td>
      <td>781</td>
      <td>fight the risk of cancer p cfm o s pk slim dow...</td>
      <td>[fight, the, risk, of, cancer, p, cfm, o, s, p...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1) Fight The Risk of Cancer!\nhttp://www.adcli...</td>
      <td>Guaranteed to lose 10-12 lbs in 30 days       ...</td>
      <td>sabrina@mx3.1premio.com</td>
      <td>[http://www.adclick.ws, http://www.adclick.ws,...</td>
      <td>[www.adclick.ws, www.adclick.ws, www.adclick.w...</td>
      <td>636</td>
      <td>fight the risk of cancer p cfm o s pk slim dow...</td>
      <td>[fight, the, risk, of, cancer, p, cfm, o, s, p...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>84762</th>
      <td>0</td>
      <td>Author: metze\nDate: 2007-04-16 07:41:12 +0000...</td>
      <td>svn commit: samba r22248 - in\n\tbranches/SAMB...</td>
      <td>metze@samba.org</td>
      <td>[http://websvn.samba.org]</td>
      <td>[websvn.samba.org]</td>
      <td>840</td>
      <td>author metze date mon apr new revision websvn ...</td>
      <td>[author, metze, date, mon, apr, new, revision,...</td>
    </tr>
    <tr>
      <th>84763</th>
      <td>0</td>
      <td>On 4/9/07, Tom Phoenix  wrote:\n&gt;\n&gt; On 4/9/07...</td>
      <td>Re: OT: html checkbox question</td>
      <td>jm5379@gmail.com</td>
      <td>0</td>
      <td>0</td>
      <td>562</td>
      <td>on tom phoenix wrote on jm wrote the first gro...</td>
      <td>[on, tom, phoenix, wrote, on, jm, wrote, the, ...</td>
    </tr>
    <tr>
      <th>84764</th>
      <td>0</td>
      <td>\n\nCharlie wrote:\n&gt; Hi, this is Charlie and ...</td>
      <td>Re: [R] Question for install Rapache package.</td>
      <td>ligges@statistik.uni-dortmund.de</td>
      <td>[https://stat.ethz.ch, http://www.R-project.or...</td>
      <td>[stat.ethz.ch, www.R-project.org, stat.ethz.ch...</td>
      <td>1178</td>
      <td>charlie wrote hi this is charlie and i am tryi...</td>
      <td>[charlie, wrote, hi, this, is, charlie, and, i...</td>
    </tr>
    <tr>
      <th>84765</th>
      <td>0</td>
      <td>Author: kseeger\nDate: 2007-04-16 07:47:27 +00...</td>
      <td>svn commit: samba-docs r1098 - in trunk: manpa...</td>
      <td>kseeger@samba.org</td>
      <td>[http://websvn.samba.org, http://www.samba.org...</td>
      <td>[websvn.samba.org, www.samba.org, www.samba.org]</td>
      <td>12536</td>
      <td>author kseeger date mon apr new revision websv...</td>
      <td>[author, kseeger, date, mon, apr, new, revisio...</td>
    </tr>
    <tr>
      <th>84766</th>
      <td>0</td>
      <td>**********************************************...</td>
      <td>CBS SportsLine Daily Sports Report</td>
      <td>mailer@mailer-relay.sportsline.com</td>
      <td>[http://ww1.sportsline.com, http://ww1.sportsl...</td>
      <td>[ww1.sportsline.com, ww1.sportsline.com, ww1.s...</td>
      <td>8518</td>
      <td>to view the color version of this message or i...</td>
      <td>[to, view, the, color, version, of, this, mess...</td>
    </tr>
  </tbody>
</table>
<p>84767 rows × 9 columns</p>
</div>




```python
df['tokens']  = df['token_text'].str.len()
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>spam</th>
      <th>raw_data</th>
      <th>subject</th>
      <th>from</th>
      <th>urls</th>
      <th>domains</th>
      <th>chars</th>
      <th>data</th>
      <th>token_text</th>
      <th>tokens</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Greetings!\n\nYou are receiving this letter be...</td>
      <td>[ILUG] STOP THE MLM INSANITY</td>
      <td>startnow2002@hotmail.com</td>
      <td>[http://www.linux.ie]</td>
      <td>[www.linux.ie]</td>
      <td>3027</td>
      <td>greetings you are receiving this letter becaus...</td>
      <td>[greetings, you, are, receiving, this, letter,...</td>
      <td>510</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>b'\n\n\n\n\n\n  \n  \n    \n      \n    \n  \n...</td>
      <td>Life Insurance - Why Pay More?</td>
      <td>12a1mailbot1@web.de</td>
      <td>[http://website.e365.cc]</td>
      <td>[website.e365.cc]</td>
      <td>1423</td>
      <td>b save up to on life insurance why spend more ...</td>
      <td>[b, save, up, to, on, life, insurance, why, sp...</td>
      <td>173</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>\n\n\n\n\n\nThe Need For Safety Is Real In 200...</td>
      <td>Real Protection, Stun Guns!  Free Shipping! Ti...</td>
      <td>lmrn@mailexcite.com</td>
      <td>[http://www.geocities.com, http://www.geocitie...</td>
      <td>[www.geocities.com, www.geocities.com, www.geo...</td>
      <td>4349</td>
      <td>the need for safety is real in you might only ...</td>
      <td>[the, need, for, safety, is, real, in, you, mi...</td>
      <td>533</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1) Fight The Risk of Cancer!\nhttp://www.adcli...</td>
      <td>[ILUG] Guaranteed to lose 10-12 lbs in 30 days...</td>
      <td>taylor@s3.serveimage.com</td>
      <td>[http://www.adclick.ws, http://www.adclick.ws,...</td>
      <td>[www.adclick.ws, www.adclick.ws, www.adclick.w...</td>
      <td>781</td>
      <td>fight the risk of cancer p cfm o s pk slim dow...</td>
      <td>[fight, the, risk, of, cancer, p, cfm, o, s, p...</td>
      <td>95</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1) Fight The Risk of Cancer!\nhttp://www.adcli...</td>
      <td>Guaranteed to lose 10-12 lbs in 30 days       ...</td>
      <td>sabrina@mx3.1premio.com</td>
      <td>[http://www.adclick.ws, http://www.adclick.ws,...</td>
      <td>[www.adclick.ws, www.adclick.ws, www.adclick.w...</td>
      <td>636</td>
      <td>fight the risk of cancer p cfm o s pk slim dow...</td>
      <td>[fight, the, risk, of, cancer, p, cfm, o, s, p...</td>
      <td>82</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>84762</th>
      <td>0</td>
      <td>Author: metze\nDate: 2007-04-16 07:41:12 +0000...</td>
      <td>svn commit: samba r22248 - in\n\tbranches/SAMB...</td>
      <td>metze@samba.org</td>
      <td>[http://websvn.samba.org]</td>
      <td>[websvn.samba.org]</td>
      <td>840</td>
      <td>author metze date mon apr new revision websvn ...</td>
      <td>[author, metze, date, mon, apr, new, revision,...</td>
      <td>88</td>
    </tr>
    <tr>
      <th>84763</th>
      <td>0</td>
      <td>On 4/9/07, Tom Phoenix  wrote:\n&gt;\n&gt; On 4/9/07...</td>
      <td>Re: OT: html checkbox question</td>
      <td>jm5379@gmail.com</td>
      <td>0</td>
      <td>0</td>
      <td>562</td>
      <td>on tom phoenix wrote on jm wrote the first gro...</td>
      <td>[on, tom, phoenix, wrote, on, jm, wrote, the, ...</td>
      <td>89</td>
    </tr>
    <tr>
      <th>84764</th>
      <td>0</td>
      <td>\n\nCharlie wrote:\n&gt; Hi, this is Charlie and ...</td>
      <td>Re: [R] Question for install Rapache package.</td>
      <td>ligges@statistik.uni-dortmund.de</td>
      <td>[https://stat.ethz.ch, http://www.R-project.or...</td>
      <td>[stat.ethz.ch, www.R-project.org, stat.ethz.ch...</td>
      <td>1178</td>
      <td>charlie wrote hi this is charlie and i am tryi...</td>
      <td>[charlie, wrote, hi, this, is, charlie, and, i...</td>
      <td>151</td>
    </tr>
    <tr>
      <th>84765</th>
      <td>0</td>
      <td>Author: kseeger\nDate: 2007-04-16 07:47:27 +00...</td>
      <td>svn commit: samba-docs r1098 - in trunk: manpa...</td>
      <td>kseeger@samba.org</td>
      <td>[http://websvn.samba.org, http://www.samba.org...</td>
      <td>[websvn.samba.org, www.samba.org, www.samba.org]</td>
      <td>12536</td>
      <td>author kseeger date mon apr new revision websv...</td>
      <td>[author, kseeger, date, mon, apr, new, revisio...</td>
      <td>1531</td>
    </tr>
    <tr>
      <th>84766</th>
      <td>0</td>
      <td>**********************************************...</td>
      <td>CBS SportsLine Daily Sports Report</td>
      <td>mailer@mailer-relay.sportsline.com</td>
      <td>[http://ww1.sportsline.com, http://ww1.sportsl...</td>
      <td>[ww1.sportsline.com, ww1.sportsline.com, ww1.s...</td>
      <td>8518</td>
      <td>to view the color version of this message or i...</td>
      <td>[to, view, the, color, version, of, this, mess...</td>
      <td>1021</td>
    </tr>
  </tbody>
</table>
<p>84767 rows × 10 columns</p>
</div>




```python
# pairplot
fg = sns.pairplot(data=df, hue="spam")
plt.show(fg)
```


    
![png](output_26_0.png)
    



```python
# export dataframe to csv
#header = ['spam', 'data', 'subject', 'from', 'auth_error', 'urls', 'domains', 'No_of_Characters', 'No_of_Words']
#df.to_csv('output.csv', columns = header)
```


```python
# nltk.download('stopwords')

def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    filtered_text = [word for word in text if word not in stop_words]
    return filtered_text
```


```python
%%time
df["stop_text"] = df["token_text"].apply(remove_stopwords)
```

    Wall time: 11.6 s
    


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>spam</th>
      <th>raw_data</th>
      <th>subject</th>
      <th>from</th>
      <th>urls</th>
      <th>domains</th>
      <th>chars</th>
      <th>data</th>
      <th>token_text</th>
      <th>tokens</th>
      <th>stop_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Greetings!\n\nYou are receiving this letter be...</td>
      <td>[ILUG] STOP THE MLM INSANITY</td>
      <td>startnow2002@hotmail.com</td>
      <td>[http://www.linux.ie]</td>
      <td>[www.linux.ie]</td>
      <td>3027</td>
      <td>greetings you are receiving this letter becaus...</td>
      <td>[greetings, you, are, receiving, this, letter,...</td>
      <td>510</td>
      <td>[greetings, receiving, letter, expressed, inte...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>b'\n\n\n\n\n\n  \n  \n    \n      \n    \n  \n...</td>
      <td>Life Insurance - Why Pay More?</td>
      <td>12a1mailbot1@web.de</td>
      <td>[http://website.e365.cc]</td>
      <td>[website.e365.cc]</td>
      <td>1423</td>
      <td>b save up to on life insurance why spend more ...</td>
      <td>[b, save, up, to, on, life, insurance, why, sp...</td>
      <td>173</td>
      <td>[b, save, life, insurance, spend, life, quote,...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>\n\n\n\n\n\nThe Need For Safety Is Real In 200...</td>
      <td>Real Protection, Stun Guns!  Free Shipping! Ti...</td>
      <td>lmrn@mailexcite.com</td>
      <td>[http://www.geocities.com, http://www.geocitie...</td>
      <td>[www.geocities.com, www.geocities.com, www.geo...</td>
      <td>4349</td>
      <td>the need for safety is real in you might only ...</td>
      <td>[the, need, for, safety, is, real, in, you, mi...</td>
      <td>533</td>
      <td>[need, safety, real, might, get, one, chance, ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1) Fight The Risk of Cancer!\nhttp://www.adcli...</td>
      <td>[ILUG] Guaranteed to lose 10-12 lbs in 30 days...</td>
      <td>taylor@s3.serveimage.com</td>
      <td>[http://www.adclick.ws, http://www.adclick.ws,...</td>
      <td>[www.adclick.ws, www.adclick.ws, www.adclick.w...</td>
      <td>781</td>
      <td>fight the risk of cancer p cfm o s pk slim dow...</td>
      <td>[fight, the, risk, of, cancer, p, cfm, o, s, p...</td>
      <td>95</td>
      <td>[fight, risk, cancer, p, cfm, pk, slim, guaran...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1) Fight The Risk of Cancer!\nhttp://www.adcli...</td>
      <td>Guaranteed to lose 10-12 lbs in 30 days       ...</td>
      <td>sabrina@mx3.1premio.com</td>
      <td>[http://www.adclick.ws, http://www.adclick.ws,...</td>
      <td>[www.adclick.ws, www.adclick.ws, www.adclick.w...</td>
      <td>636</td>
      <td>fight the risk of cancer p cfm o s pk slim dow...</td>
      <td>[fight, the, risk, of, cancer, p, cfm, o, s, p...</td>
      <td>82</td>
      <td>[fight, risk, cancer, p, cfm, pk, slim, guaran...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>84762</th>
      <td>0</td>
      <td>Author: metze\nDate: 2007-04-16 07:41:12 +0000...</td>
      <td>svn commit: samba r22248 - in\n\tbranches/SAMB...</td>
      <td>metze@samba.org</td>
      <td>[http://websvn.samba.org]</td>
      <td>[websvn.samba.org]</td>
      <td>840</td>
      <td>author metze date mon apr new revision websvn ...</td>
      <td>[author, metze, date, mon, apr, new, revision,...</td>
      <td>88</td>
      <td>[author, metze, date, mon, apr, new, revision,...</td>
    </tr>
    <tr>
      <th>84763</th>
      <td>0</td>
      <td>On 4/9/07, Tom Phoenix  wrote:\n&gt;\n&gt; On 4/9/07...</td>
      <td>Re: OT: html checkbox question</td>
      <td>jm5379@gmail.com</td>
      <td>0</td>
      <td>0</td>
      <td>562</td>
      <td>on tom phoenix wrote on jm wrote the first gro...</td>
      <td>[on, tom, phoenix, wrote, on, jm, wrote, the, ...</td>
      <td>89</td>
      <td>[tom, phoenix, wrote, jm, wrote, first, group,...</td>
    </tr>
    <tr>
      <th>84764</th>
      <td>0</td>
      <td>\n\nCharlie wrote:\n&gt; Hi, this is Charlie and ...</td>
      <td>Re: [R] Question for install Rapache package.</td>
      <td>ligges@statistik.uni-dortmund.de</td>
      <td>[https://stat.ethz.ch, http://www.R-project.or...</td>
      <td>[stat.ethz.ch, www.R-project.org, stat.ethz.ch...</td>
      <td>1178</td>
      <td>charlie wrote hi this is charlie and i am tryi...</td>
      <td>[charlie, wrote, hi, this, is, charlie, and, i...</td>
      <td>151</td>
      <td>[charlie, wrote, hi, charlie, trying, embed, r...</td>
    </tr>
    <tr>
      <th>84765</th>
      <td>0</td>
      <td>Author: kseeger\nDate: 2007-04-16 07:47:27 +00...</td>
      <td>svn commit: samba-docs r1098 - in trunk: manpa...</td>
      <td>kseeger@samba.org</td>
      <td>[http://websvn.samba.org, http://www.samba.org...</td>
      <td>[websvn.samba.org, www.samba.org, www.samba.org]</td>
      <td>12536</td>
      <td>author kseeger date mon apr new revision websv...</td>
      <td>[author, kseeger, date, mon, apr, new, revisio...</td>
      <td>1531</td>
      <td>[author, kseeger, date, mon, apr, new, revisio...</td>
    </tr>
    <tr>
      <th>84766</th>
      <td>0</td>
      <td>**********************************************...</td>
      <td>CBS SportsLine Daily Sports Report</td>
      <td>mailer@mailer-relay.sportsline.com</td>
      <td>[http://ww1.sportsline.com, http://ww1.sportsl...</td>
      <td>[ww1.sportsline.com, ww1.sportsline.com, ww1.s...</td>
      <td>8518</td>
      <td>to view the color version of this message or i...</td>
      <td>[to, view, the, color, version, of, this, mess...</td>
      <td>1021</td>
      <td>[view, color, version, message, links, work, p...</td>
    </tr>
  </tbody>
</table>
<p>84767 rows × 11 columns</p>
</div>




```python
#nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def lemmatize_word(text):
    lemmas = [lemmatizer.lemmatize(word, pos ='v') for word in text]
    return lemmas
```


```python
%%time
df["clean_text"] = df["stop_text"].apply(lemmatize_word)
```

    Wall time: 27.5 s
    


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>spam</th>
      <th>raw_data</th>
      <th>subject</th>
      <th>from</th>
      <th>urls</th>
      <th>domains</th>
      <th>chars</th>
      <th>data</th>
      <th>token_text</th>
      <th>tokens</th>
      <th>stop_text</th>
      <th>clean_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Greetings!\n\nYou are receiving this letter be...</td>
      <td>[ILUG] STOP THE MLM INSANITY</td>
      <td>startnow2002@hotmail.com</td>
      <td>[http://www.linux.ie]</td>
      <td>[www.linux.ie]</td>
      <td>3027</td>
      <td>greetings you are receiving this letter becaus...</td>
      <td>[greetings, you, are, receiving, this, letter,...</td>
      <td>510</td>
      <td>[greetings, receiving, letter, expressed, inte...</td>
      <td>[greet, receive, letter, express, interest, re...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>b'\n\n\n\n\n\n  \n  \n    \n      \n    \n  \n...</td>
      <td>Life Insurance - Why Pay More?</td>
      <td>12a1mailbot1@web.de</td>
      <td>[http://website.e365.cc]</td>
      <td>[website.e365.cc]</td>
      <td>1423</td>
      <td>b save up to on life insurance why spend more ...</td>
      <td>[b, save, up, to, on, life, insurance, why, sp...</td>
      <td>173</td>
      <td>[b, save, life, insurance, spend, life, quote,...</td>
      <td>[b, save, life, insurance, spend, life, quote,...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>\n\n\n\n\n\nThe Need For Safety Is Real In 200...</td>
      <td>Real Protection, Stun Guns!  Free Shipping! Ti...</td>
      <td>lmrn@mailexcite.com</td>
      <td>[http://www.geocities.com, http://www.geocitie...</td>
      <td>[www.geocities.com, www.geocities.com, www.geo...</td>
      <td>4349</td>
      <td>the need for safety is real in you might only ...</td>
      <td>[the, need, for, safety, is, real, in, you, mi...</td>
      <td>533</td>
      <td>[need, safety, real, might, get, one, chance, ...</td>
      <td>[need, safety, real, might, get, one, chance, ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1) Fight The Risk of Cancer!\nhttp://www.adcli...</td>
      <td>[ILUG] Guaranteed to lose 10-12 lbs in 30 days...</td>
      <td>taylor@s3.serveimage.com</td>
      <td>[http://www.adclick.ws, http://www.adclick.ws,...</td>
      <td>[www.adclick.ws, www.adclick.ws, www.adclick.w...</td>
      <td>781</td>
      <td>fight the risk of cancer p cfm o s pk slim dow...</td>
      <td>[fight, the, risk, of, cancer, p, cfm, o, s, p...</td>
      <td>95</td>
      <td>[fight, risk, cancer, p, cfm, pk, slim, guaran...</td>
      <td>[fight, risk, cancer, p, cfm, pk, slim, guaran...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1) Fight The Risk of Cancer!\nhttp://www.adcli...</td>
      <td>Guaranteed to lose 10-12 lbs in 30 days       ...</td>
      <td>sabrina@mx3.1premio.com</td>
      <td>[http://www.adclick.ws, http://www.adclick.ws,...</td>
      <td>[www.adclick.ws, www.adclick.ws, www.adclick.w...</td>
      <td>636</td>
      <td>fight the risk of cancer p cfm o s pk slim dow...</td>
      <td>[fight, the, risk, of, cancer, p, cfm, o, s, p...</td>
      <td>82</td>
      <td>[fight, risk, cancer, p, cfm, pk, slim, guaran...</td>
      <td>[fight, risk, cancer, p, cfm, pk, slim, guaran...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>84762</th>
      <td>0</td>
      <td>Author: metze\nDate: 2007-04-16 07:41:12 +0000...</td>
      <td>svn commit: samba r22248 - in\n\tbranches/SAMB...</td>
      <td>metze@samba.org</td>
      <td>[http://websvn.samba.org]</td>
      <td>[websvn.samba.org]</td>
      <td>840</td>
      <td>author metze date mon apr new revision websvn ...</td>
      <td>[author, metze, date, mon, apr, new, revision,...</td>
      <td>88</td>
      <td>[author, metze, date, mon, apr, new, revision,...</td>
      <td>[author, metze, date, mon, apr, new, revision,...</td>
    </tr>
    <tr>
      <th>84763</th>
      <td>0</td>
      <td>On 4/9/07, Tom Phoenix  wrote:\n&gt;\n&gt; On 4/9/07...</td>
      <td>Re: OT: html checkbox question</td>
      <td>jm5379@gmail.com</td>
      <td>0</td>
      <td>0</td>
      <td>562</td>
      <td>on tom phoenix wrote on jm wrote the first gro...</td>
      <td>[on, tom, phoenix, wrote, on, jm, wrote, the, ...</td>
      <td>89</td>
      <td>[tom, phoenix, wrote, jm, wrote, first, group,...</td>
      <td>[tom, phoenix, write, jm, write, first, group,...</td>
    </tr>
    <tr>
      <th>84764</th>
      <td>0</td>
      <td>\n\nCharlie wrote:\n&gt; Hi, this is Charlie and ...</td>
      <td>Re: [R] Question for install Rapache package.</td>
      <td>ligges@statistik.uni-dortmund.de</td>
      <td>[https://stat.ethz.ch, http://www.R-project.or...</td>
      <td>[stat.ethz.ch, www.R-project.org, stat.ethz.ch...</td>
      <td>1178</td>
      <td>charlie wrote hi this is charlie and i am tryi...</td>
      <td>[charlie, wrote, hi, this, is, charlie, and, i...</td>
      <td>151</td>
      <td>[charlie, wrote, hi, charlie, trying, embed, r...</td>
      <td>[charlie, write, hi, charlie, try, embed, r, a...</td>
    </tr>
    <tr>
      <th>84765</th>
      <td>0</td>
      <td>Author: kseeger\nDate: 2007-04-16 07:47:27 +00...</td>
      <td>svn commit: samba-docs r1098 - in trunk: manpa...</td>
      <td>kseeger@samba.org</td>
      <td>[http://websvn.samba.org, http://www.samba.org...</td>
      <td>[websvn.samba.org, www.samba.org, www.samba.org]</td>
      <td>12536</td>
      <td>author kseeger date mon apr new revision websv...</td>
      <td>[author, kseeger, date, mon, apr, new, revisio...</td>
      <td>1531</td>
      <td>[author, kseeger, date, mon, apr, new, revisio...</td>
      <td>[author, kseeger, date, mon, apr, new, revisio...</td>
    </tr>
    <tr>
      <th>84766</th>
      <td>0</td>
      <td>**********************************************...</td>
      <td>CBS SportsLine Daily Sports Report</td>
      <td>mailer@mailer-relay.sportsline.com</td>
      <td>[http://ww1.sportsline.com, http://ww1.sportsl...</td>
      <td>[ww1.sportsline.com, ww1.sportsline.com, ww1.s...</td>
      <td>8518</td>
      <td>to view the color version of this message or i...</td>
      <td>[to, view, the, color, version, of, this, mess...</td>
      <td>1021</td>
      <td>[view, color, version, message, links, work, p...</td>
      <td>[view, color, version, message, link, work, pr...</td>
    </tr>
  </tbody>
</table>
<p>84767 rows × 12 columns</p>
</div>




```python
## CREATE DB
import sqlite3

con = sqlite3.connect('bad_domains.db')
cur = con.cursor()

cur.execute("CREATE TABLE IF NOT EXISTS domains (domain TEXT PRIMARY KEY);")


with open("/resources/bad-domains.txt", "r") as f:
    for line in f.readlines():
        if len(line) == 0:
            continue
        line = line.replace('\n', '')
        try:        
            cur.execute("INSERT INTO domains VALUES ('%s')" % line)
        except:
            pass
    
con.commit()
con.close()
```


```python
def check_domain(domains):
    
    if not domains:
        return 0
    
    con = sqlite3.connect('resources/bad_domains.db')
    cur = con.cursor()
    found = 0
    
    for domain in domains:
        cur.execute("SELECT * FROM domains WHERE domain = '%s'" % domain)
        if len(cur.fetchall()) > 0:
            found = 1
            break

    con.close()
    return found
```


```python
%%time
df["bad_domain"] = df["domains"].apply(check_domain)
```

    Wall time: 30.4 s
    


```python
df["bad_domain"]
```




    0        0
    1        0
    2        0
    3        0
    4        0
            ..
    84762    0
    84763    0
    84764    0
    84765    0
    84766    0
    Name: bad_domain, Length: 84767, dtype: int64




```python
df.groupby(['spam', 'bad_domain']).size()
```




    spam  bad_domain
    0     0             31358
          1               813
    1     0             52453
          1               143
    dtype: int64




```python
sns.countplot(df.bad_domain)
```

    c:\users\konstantin\appdata\local\programs\python\python39\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    




    <AxesSubplot:xlabel='bad_domain', ylabel='count'>




    
![png](output_39_2.png)
    



```python
# there are more blacklisted urls in ham mails than in spam mails --> do not use this feature
```


```python
def contains_urls(url_list):
    if url_list == 0:
        return 0
    else:
        return 1
```


```python
df["contains_urls"] = df["domains"].apply(contains_urls)
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>spam</th>
      <th>raw_data</th>
      <th>subject</th>
      <th>from</th>
      <th>urls</th>
      <th>domains</th>
      <th>chars</th>
      <th>data</th>
      <th>token_text</th>
      <th>tokens</th>
      <th>stop_text</th>
      <th>clean_text</th>
      <th>bad_domain</th>
      <th>contains_urls</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Greetings!\n\nYou are receiving this letter be...</td>
      <td>[ILUG] STOP THE MLM INSANITY</td>
      <td>startnow2002@hotmail.com</td>
      <td>[http://www.linux.ie]</td>
      <td>[www.linux.ie]</td>
      <td>3027</td>
      <td>greetings you are receiving this letter becaus...</td>
      <td>[greetings, you, are, receiving, this, letter,...</td>
      <td>510</td>
      <td>[greetings, receiving, letter, expressed, inte...</td>
      <td>[greet, receive, letter, express, interest, re...</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>b'\n\n\n\n\n\n  \n  \n    \n      \n    \n  \n...</td>
      <td>Life Insurance - Why Pay More?</td>
      <td>12a1mailbot1@web.de</td>
      <td>[http://website.e365.cc]</td>
      <td>[website.e365.cc]</td>
      <td>1423</td>
      <td>b save up to on life insurance why spend more ...</td>
      <td>[b, save, up, to, on, life, insurance, why, sp...</td>
      <td>173</td>
      <td>[b, save, life, insurance, spend, life, quote,...</td>
      <td>[b, save, life, insurance, spend, life, quote,...</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>\n\n\n\n\n\nThe Need For Safety Is Real In 200...</td>
      <td>Real Protection, Stun Guns!  Free Shipping! Ti...</td>
      <td>lmrn@mailexcite.com</td>
      <td>[http://www.geocities.com, http://www.geocitie...</td>
      <td>[www.geocities.com, www.geocities.com, www.geo...</td>
      <td>4349</td>
      <td>the need for safety is real in you might only ...</td>
      <td>[the, need, for, safety, is, real, in, you, mi...</td>
      <td>533</td>
      <td>[need, safety, real, might, get, one, chance, ...</td>
      <td>[need, safety, real, might, get, one, chance, ...</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1) Fight The Risk of Cancer!\nhttp://www.adcli...</td>
      <td>[ILUG] Guaranteed to lose 10-12 lbs in 30 days...</td>
      <td>taylor@s3.serveimage.com</td>
      <td>[http://www.adclick.ws, http://www.adclick.ws,...</td>
      <td>[www.adclick.ws, www.adclick.ws, www.adclick.w...</td>
      <td>781</td>
      <td>fight the risk of cancer p cfm o s pk slim dow...</td>
      <td>[fight, the, risk, of, cancer, p, cfm, o, s, p...</td>
      <td>95</td>
      <td>[fight, risk, cancer, p, cfm, pk, slim, guaran...</td>
      <td>[fight, risk, cancer, p, cfm, pk, slim, guaran...</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1) Fight The Risk of Cancer!\nhttp://www.adcli...</td>
      <td>Guaranteed to lose 10-12 lbs in 30 days       ...</td>
      <td>sabrina@mx3.1premio.com</td>
      <td>[http://www.adclick.ws, http://www.adclick.ws,...</td>
      <td>[www.adclick.ws, www.adclick.ws, www.adclick.w...</td>
      <td>636</td>
      <td>fight the risk of cancer p cfm o s pk slim dow...</td>
      <td>[fight, the, risk, of, cancer, p, cfm, o, s, p...</td>
      <td>82</td>
      <td>[fight, risk, cancer, p, cfm, pk, slim, guaran...</td>
      <td>[fight, risk, cancer, p, cfm, pk, slim, guaran...</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>84762</th>
      <td>0</td>
      <td>Author: metze\nDate: 2007-04-16 07:41:12 +0000...</td>
      <td>svn commit: samba r22248 - in\n\tbranches/SAMB...</td>
      <td>metze@samba.org</td>
      <td>[http://websvn.samba.org]</td>
      <td>[websvn.samba.org]</td>
      <td>840</td>
      <td>author metze date mon apr new revision websvn ...</td>
      <td>[author, metze, date, mon, apr, new, revision,...</td>
      <td>88</td>
      <td>[author, metze, date, mon, apr, new, revision,...</td>
      <td>[author, metze, date, mon, apr, new, revision,...</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>84763</th>
      <td>0</td>
      <td>On 4/9/07, Tom Phoenix  wrote:\n&gt;\n&gt; On 4/9/07...</td>
      <td>Re: OT: html checkbox question</td>
      <td>jm5379@gmail.com</td>
      <td>0</td>
      <td>0</td>
      <td>562</td>
      <td>on tom phoenix wrote on jm wrote the first gro...</td>
      <td>[on, tom, phoenix, wrote, on, jm, wrote, the, ...</td>
      <td>89</td>
      <td>[tom, phoenix, wrote, jm, wrote, first, group,...</td>
      <td>[tom, phoenix, write, jm, write, first, group,...</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>84764</th>
      <td>0</td>
      <td>\n\nCharlie wrote:\n&gt; Hi, this is Charlie and ...</td>
      <td>Re: [R] Question for install Rapache package.</td>
      <td>ligges@statistik.uni-dortmund.de</td>
      <td>[https://stat.ethz.ch, http://www.R-project.or...</td>
      <td>[stat.ethz.ch, www.R-project.org, stat.ethz.ch...</td>
      <td>1178</td>
      <td>charlie wrote hi this is charlie and i am tryi...</td>
      <td>[charlie, wrote, hi, this, is, charlie, and, i...</td>
      <td>151</td>
      <td>[charlie, wrote, hi, charlie, trying, embed, r...</td>
      <td>[charlie, write, hi, charlie, try, embed, r, a...</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>84765</th>
      <td>0</td>
      <td>Author: kseeger\nDate: 2007-04-16 07:47:27 +00...</td>
      <td>svn commit: samba-docs r1098 - in trunk: manpa...</td>
      <td>kseeger@samba.org</td>
      <td>[http://websvn.samba.org, http://www.samba.org...</td>
      <td>[websvn.samba.org, www.samba.org, www.samba.org]</td>
      <td>12536</td>
      <td>author kseeger date mon apr new revision websv...</td>
      <td>[author, kseeger, date, mon, apr, new, revisio...</td>
      <td>1531</td>
      <td>[author, kseeger, date, mon, apr, new, revisio...</td>
      <td>[author, kseeger, date, mon, apr, new, revisio...</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>84766</th>
      <td>0</td>
      <td>**********************************************...</td>
      <td>CBS SportsLine Daily Sports Report</td>
      <td>mailer@mailer-relay.sportsline.com</td>
      <td>[http://ww1.sportsline.com, http://ww1.sportsl...</td>
      <td>[ww1.sportsline.com, ww1.sportsline.com, ww1.s...</td>
      <td>8518</td>
      <td>to view the color version of this message or i...</td>
      <td>[to, view, the, color, version, of, this, mess...</td>
      <td>1021</td>
      <td>[view, color, version, message, links, work, p...</td>
      <td>[view, color, version, message, link, work, pr...</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>84767 rows × 14 columns</p>
</div>




```python
# Remove duplicates
df = df.drop_duplicates(subset="raw_data", keep="first")

# Remove emails with less then five words
print(f"All emails without dups: {len(df)}")
print(f"Emails with less then five words: {len(df[df['clean_text'].map(lambda d: len(d)) < 5])}")
df = df.drop(df[df['clean_text'].map(lambda d: len(d)) < 5].index)
print(f"Emails with at least 5 words: {len(df)}")
```

    All emails without dups: 63570
    Emails with less then five words: 869
    Emails with at least 5 words: 62701
    


```python
# dataframe without dups
joblib.dump(df, "exports/dataframe.sav")
```




    ['dataframe.sav']


