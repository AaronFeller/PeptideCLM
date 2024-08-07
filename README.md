# PeptideCLM

## Introduction

PeptideCLM-23.4M is a peptide-specific pretrained chemical language model published in [future publication]. 
The base model was pretrained with masked language modeling on 23.4 million molecules, divided approximately into two halves: small molecules and peptides.
Finetuned models are trained to predict membrane penetration for cyclic peptides from CycPeptMPDB.

This repository contains some example code for loading pre-trained weights and notebooks with analysis conducted for assessment of the model. 

## Installation

Required libraries for using this model include the Transformers library with its dependencies.


All model architecture and weights are hosted on [huggingface](https://huggingface.co/aaronfeller).
You can load them using the following code:
```
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(model_name) 
```

## All models on huggingface
|    Name    | `aaronfeller.`     | Dataset | Description
| Pretrained PeptideCLM | `PeptideCLM-23.4M-all` | PubChem + SureChEMBL + SmProt + RandPept | Fully pretrained PeptideCLM which performed best in finetuning on the downstream task of predicting peptide membrane penetration. |
|            | `PeptideCLM-12.6M-smol` | PubChem + SureChEMBL | PeptideCLM trained on the small molecule portion of pretraining data. |
|            | `PeptideCLM-10.8M-pep` | SmProt + RandPept | PeptideCLM trained on the peptide portion of pretraining data. |
| Finetuned PeptideCLM |  `PeptideCLM-23.4M-CycPeptMPDB-fold-1` | CycPeptMPDB | Five models below are finetuned on a subset of PAMPA data from CycPeptMPDB with 5-fold cross validation on k-means clustered embeddings. |
|                      |  `PeptideCLM-23.4M-CycPeptMPDB-fold-2` | | |
|                      |  `PeptideCLM-23.4M-CycPeptMPDB-fold-3` | | |
|                      |  `PeptideCLM-23.4M-CycPeptMPDB-fold-4` | | |
|                      |  `PeptideCLM-23.4M-CycPeptMPDB-fold-5` | | |

