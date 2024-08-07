# PeptideCLM

## Introduction

PeptideCLM-23.4M is a peptide-specific pretrained chemical language model published in [future publication]. 
The base model was pretrained with masked language modeling on 23.4 million molecules, divided approximately into two halves: small molecules and peptides.
Finetuned models are trained to predict membrane penetration for cyclic peptides from CycPeptMPDB.

This repository contains some example code for loading pre-trained weights and notebooks with analysis conducted for assessment of the model. 

## Installation

Required libraries for using this model include:
```
transformers
pytorch
```

All model architecture and weights are hosted on [huggingface](https://huggingface.co/aaronfeller).
