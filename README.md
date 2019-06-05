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
The naive baseline of Single Task Model (**STM** in paper) followed [NCRF++](https://github.com/jiesutd/NCRFpp
), with some exceptions in the hyperparameters of STM-Target in order to give a strong baseline. 
<br> <br>
For more details, please refer to our paper "[Cross-Domain NER using Cross-Domain Language Modeling]()".

## Requirements
```
Python 2 or 3 
PyTorch 0.3
```
The cache memory of one GPU should no less than 8GB.

## Pretrained Embeddings
GloVe 100-dimension word vectors (Cite from [*Here*](https://www.aclweb.org/anthology/D14-1162)).

## Data
### Labeled data
* Source-domain: CoNLL-2003 English NER data. <br>
* Target-domain
 * Unsupervised: CBS SciTech News (test set).(In `\combined_SDA_and_UDA\data\tech_test`) <br>
 * Supervised: [BioNLP13PC](https://github.com/cambridgeltl/MTL-Bioinformatics-2016), and [BioNLP13CG](https://github.com/cambridgeltl/MTL-Bioinformatics-2016)

### Raw data
* Source-domain: 377,592 sentences from the Reuters [*Download*](https://pan.baidu.com/s/1Sl5JssWV8R18nTU6S3Brrw) with a key `r12a`.
* Target-domain(unsupervised): 398,990 sentences from CBS SciTech News [*Download*](https://pan.baidu.com/s/1CGBWuf5XTfFmimXmLTBFwA) with a key `7w5h`.
* Optional Biomedicine raw data from the PubMed can also be tried [*Download*](https://pan.baidu.com/s/1s866FUl07L96JmzelMC2xw) with a key `5ijl`.

## Usage
### Command
Both `\supervised_domain_adaptation` and `\combined_SDA_and_UDA` can use the following command to make it run. <br>
```
python main.py --config train.NER.config
```
### Input format
* We recommand using the IBOES label style for NER dataset.
* We recommand using a input style of one-sentence-per-line for raw data with word segmentatian.
## Cite:
If you use our data or code, please cite our paper as follows:
```

```

## Update
* V2. Combining supervised model and unsupervised model in `\combined_SDA_and_UDA`.
* V1. The previous multi-task version in `\supervised_domain_adaptation`.
