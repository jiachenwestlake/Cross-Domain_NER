# Cross-Domain_NER
Cross-domain NER using cross-domain language modeling, code for ACL 2019 paper.

## Introduction
NER is a fundamental task in NLP. Due to the limitation of labeled resources, cross-domain NER has been a challenging task. Most previous work concentrates on the supervised scenario, making use of labeled data for both source and target domains. A disadvantage of such setting is that they can not train for domains which have no labeled data.
<br> <br>
We address this issue, using  cross-domain LM as a bridge cross-domains for NER domain adaptation. Performing cross-task and cross-domain transfer by designing a novel **Parameter Generation Network**. 
<br> <br>
Experiments on **CBS SciTech News Dataset** show that our model can effectively allow unsupervised domain adaptation,
while also can deriving supervised domain adaption between domains with completely different entity types (i.e. news vs. biomedical). 
<br> <br>
The naive baseline of Single Task Model (**STM** in paper) mostly followed [NCRF++](https://github.com/jiesutd/NCRFpp
).
<br> <br>
For more details, please refer to our paper:
<br><br>
[Cross-Domain NER using Cross-Domain Language Modeling](https://www.aclweb.org/anthology/P19-1236)
<br>
Chen Jia, Xiaobo Liang and Yue Zhang*
<br>
(* Corresponding Author)
<br>
ACL 2019

## Requirements
```
Python 2 or 3 
PyTorch 0.3
```
The memory of one GPU should be no less than 8GB to fit the model.

## Pretrained Embeddings
GloVe 100-dimension word vectors (Cite from [*Here*](https://www.aclweb.org/anthology/D14-1162)).

## DataSet
### Source-domain: 
CoNLL-2003 English NER data.
### Target-domain
 * Unsupervised: CBS SciTech News (test set) (In: `\unsupervised_domain_adaptation\data\news_tech\tech_test). <br>
 * Supervised: [BioNLP13PC](https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data) dataset and [BioNLP13CG](https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data) dataset.

## Usage
### Command
`\supervised_domain_adaptation`, `\unsupervised_domain_adaptation`and `\combined_SDA_and_UDA` can use the following command to make it run. <br>
```
python main.py --config train.NER.config
```
The file `train.NER.config` contains dataset path and model hyperparameters following [NCRF++](https://github.com/jiesutd/NCRFpp
).
### Input format
* We recommand using the IBOES label style for NER dataset.
* We recommand using an input style of one-sentence-per-line for raw data with word segmentation.
## Cite:
If you use our data or code, please cite our paper as follows:
```
@inproceedings{jia2019cross,
  title={Cross-domain ner using cross-domain language modeling},
  author={Jia, Chen and Liang, Xiaobo and Zhang, Yue},
  booktitle={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  pages={2464--2474},
  year={2019}
  organization={Association for Computational Linguistics}
}
```

## Update
* V2. Combining supervised scenario and unsupervised scenario in `\combined_SDA_and_UDA`.
* V1. The previous supervised scenario in `\supervised_domain_adaptation`; <br>
      The previous unsupervised scenario in `\unsupervised_domain_adaptation`;
