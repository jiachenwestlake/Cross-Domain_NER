# Cross-Domain_NER
Cross-domain NER using cross-domain language modeling, code for ACL 2019 paper.

## Introduction
NER is a fundamental task in NLP. Due to the limitation of labeled resources, cross-domain NER has been a challenging task. Most previous work concentrates on the supervised scenario, making use of labeled data for both source and target domains. A disadvantage of such setting is that they can not train for domains which have no labeled data.
<br> <br>
We address this issue, using  cross-domain LM as a bridge cross-domains for NER domain adaptation. Performing cross-task and cross-domain transfer by designing a novel **Parameter Generation Network**. 
<br> <br>
Experiments on **CBS SciTech News data** show that our model can effectively allow unsupervised domain adaptation,
while also can deriving supervised domain adaption between domains with completely different entity types (i.e. news vs. biomedical). 
<br> <br>
The naive baseline of Single Task Model (STM-Target and STM-Source in paper) followed [NCRF++](https://github.com/jiesutd/NCRFpp
), with some exceptions in hyperparameters of STM-Target to give a strong baseline. 
<br>
For more details, please refer to our paper "[Cross-Domain NER using Cross-Domain Language Modeling]()".

## Requirements
```
Python 2 or 3 
PyTorch 0.3
```
The cache memory of one GPU should no less than 8GB.

## Pretrained Embeddings
GloVe 100-dimension word vectors.

## Data
### Labeled data
* Source-domain: CoNLL 2003 English NER data. <br>
* Target-domain
  * Unsupervised: CBS SciTech News (test set).(In: `\combined_SDA_and_UDA\data\tech_test`) <br>
  * Supervised: [BioNLP13PC](https://github.com/cambridgeltl/MTL-Bioinformatics-2016), [BioNLP13CG](https://github.com/cambridgeltl/MTL-Bioinformatics-2016)

### Raw data
* Source-domain: 377,592 sentences from the Reuters. <br>
* Target-domain(unsupervised): 398,990 sentences from CBS SciTech News.


## Usage
Both `\supervised_domain_adaptation` and `\combined_SDA_and_UDA` can use the following command to make it run. <br>
<br>
```
python main.py --config train.NER.config
```

## Cite:

## Update
V2. Combining supervised model and unsupervised model in `\combined_SDA_and_UDA`.
<br>
V1. The formal multi-task version in `\supervised_domain_adaptation`.
