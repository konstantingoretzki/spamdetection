# spamdetection
Project repository for the spam detection Security Analytics (SECANA) project.

## description
The goal of our project was to be able to classify a email and tell if it's spam or ham (not spam) by analysing the content of the emails (natural language processing - NLP).
After processing and exploring the datasets we've compared different features and classifiers.
We've decided to use the RandomForest classifier with TFIDF (parsed and preprocessed texts of the email bodys) and the number of chars and tokens as features.

Besides the model we've also developed a web interface for checking your emails (simply upload the mail as an .eml file) and a Mozilla Thunderbird plugin that can make use of the specified web interface and provides in-app feedback. More information about these projects can be found in their linked repo.

## dataset info
We've used the following datasets:
- [Apache SpamAssassin](https://spamassassin.apache.org/old/publiccorpus/)
- [2007 TREC Public Spam Corpus](https://plg.uwaterloo.ca/cgi-bin/cgiwrap/gvcormac/foo07)

At least the TREC dataset can (due to licensing) not be uploaded so please download the datasets on your own.

The directories should look like this:
```
datasets/
├── spamassassin (files not provided)
│   ├── ham [6951 entries exceeds filelimit, not opening dir]
│   └── spam [2397 entries exceeds filelimit, not opening dir]
└── trec07p (files not provided)
│   ├── ham [25220 entries exceeds filelimit, not opening dir]
│   ├── spam [50199 entries exceeds filelimit, not opening dir]
└── splitTRECFiles.sh - helper for splitting the trec dataset files
```

Keep in mind that there might be some files in the datasets that have to be removed, e.g. READMEs or files that contain the copy-commands.

## project structure
For just training the model from just the datasets (spam and ham files) run the file `spam-detection.ipynb`.
Design decisions and plots can be found in the files `preprocessing.ipynb`, `features.ipynb`, `vocabulary.ipynb` and `model.ipynb`.

```
spam-detection/
├── LICENSE
├── README.md - you are here
├── datasets
	├── spamassassin (files not provided)
	├── trec07p (files not provided)
	├── splitTRECFiles.sh - helper for splitting the trec dataset files
├── docu
│   └── introduction - concept and goal of our topic
│       ├── SECANA-E-Mail-Klassifizierung-Gruppe-5.pdf
│       ├── data-ham.png
│       ├── data-spam.png
│       └── document.tex
├── exports
│   ├── model.sav
│   └── vocab.sav
├── features.ipynb - details about features (ideas and importances) and outliers
├── markdown - md export of the main file that contains only the necessary steps to train the model
│   └── spam-detection.md
├── model.ipynb - decisions for using the final classifier
├── preprocessing.ipynb - preprocessing incl. design thoughts and canceled ideas
├── resources - 
│   ├── bad_domains.txt - downloaded content of https://dbl.oisd.nl/ , remove headers (not provided)
│   ├── bad_domains.db - created via preprocessing.ipynb from bad_domains.txt (not provided)
│   └── words-dwyl-github.txt - https://github.com/dwyl/english-words/blob/master/words.txt (not provided)
├── vocabulary.ipynb - design decision for the vocabulary (using intersections)
└── spam-detection.ipynb - main jupyter notebook that contains only the necessary steps to train the model
```