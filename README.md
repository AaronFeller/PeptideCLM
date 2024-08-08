# PeptideCLM

## Introduction

PeptideCLM-23.4M is a peptide-specific pretrained chemical language model published in [future publication]. 
The base model was pretrained with masked language modeling on 23 million molecules, divided approximately into two halves: 12M small molecules and 11M peptides.
Finetuned models are trained to predict membrane penetration for cyclic peptides from CycPeptMPDB.

This repository contains some example code for loading pre-trained weights, which include a Jupyter notebook example of model assessment and a python script for finetuning the pretrained model. 

## Models
Required libraries for loading this model include the Transformers library with its dependencies (PyTorch, etc.) and the tokenizer library SmilesPE along with its dependencies.


| Model name              | Training dataset                                          | Description                                                                                                               |
|-----------------------------|--------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| `PeptideCLM-23M-all`         | PubChem + SureChEMBL + SmProt + RandPept | Fully pretrained PeptideCLM which performed best in finetuning on the downstream task of predicting peptide membrane penetration. |
| `PeptideCLM-12M-smol`        | PubChem + SureChEMBL                     | PeptideCLM pretrained on the small molecule portion of pretraining data. |
| `PeptideCLM-11M-pep`         | SmProt + RandPept                        | PeptideCLM pretrained on the peptide portion of pretraining data. |
| `PeptideCLM-23M-CycPeptMPDB-fold-1` | Finetuning: CycPeptMPDB | PeptideCLM-all finetuned on CycPeptMPDB PAMPA data, fold 1. |
| `PeptideCLM-23M-CycPeptMPDB-fold-2` | Finetuning: CycPeptMPDB | PeptideCLM-all finetuned on CycPeptMPDB PAMPA data, fold 2. |
| `PeptideCLM-23M-CycPeptMPDB-fold-3` | Finetuning: CycPeptMPDB | PeptideCLM-all finetuned on CycPeptMPDB PAMPA data, fold 3. |
| `PeptideCLM-23M-CycPeptMPDB-fold-4` | Finetuning: CycPeptMPDB | PeptideCLM-all finetuned on CycPeptMPDB PAMPA data, fold 4. |
| `PeptideCLM-23M-CycPeptMPDB-fold-5` | Finetuning: CycPeptMPDB | PeptideCLM-all finetuned on CycPeptMPDB PAMPA data, fold 5. |


All models hosted on [huggingface](https://huggingface.co/aaronfeller) can be loaded from [huggingface.co/aaronfeller](https://huggingface.co/aaronfeller) using the following code:
```
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(model_name) 
```

## Tokenizer

A custom tokenizer was both inspired by and built using [Smiles Pair Encoding](https://github.com/XinhaoLi74/SmilesPE).

I attempted to port my custom tokenizer to HuggingFace, but was unable to. If anyone can sort this out, I'm happy to accept a pull request either here or to HuggingFace. For now, you can import the tokenizer as shown in either examples contained in this directory.
