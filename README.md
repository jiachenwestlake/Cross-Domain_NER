# Cross-Domain_NER

## Introduction
Cross-domain NER using cross-domain language modeling, code for ACL 2019 paper. <br>
<br>
NER is a fundamental task in NLP. Due to the limitation of labeled resources, cross-domain NER has been a challenging task. Most previous work concentrates on the supervised scenario, making use of labeled data for both source and target domains. A disadvantage of such setting is that they can not train for domains which have no labeled data. <br>
<br>
We address this issue, using  cross-domain LM as a bridge cross-domains for NER domain adaptation. Performing cross-task and cross-domain transfer by designing a novel `Parameter Generation Network`. <br>
<br>
Experiments on `CBS SciTech News data` show that our model can effectively allow unsupervised domain adaptation,
while also can deriving supervised domain adaption between domains with completely different entity types (i.e. news vs. biomedical).


## Requirements
Python 2 or 3 <br>
PyTorch 0.3 


## Pretrained Embeddings
GloVe 100-dimension word vectors

## Data
### Labeled data
Source-domain: CoNLL 2003 English NER data (train, dev, and test sets). <br>
Target-domain(unsupervised): CBS SciTech News (test set).
### Raw data
Source-domain: 377,592 sentences from the Reuters. <br>
Target-domain: 398,990 sentences from CBS SciTech News.


## Usage


## Cite:
