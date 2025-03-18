# PeptideCLM
This work was developed in the [Wilke lab](https://wilkelab.org/) by Aaron Feller, as part of his PhD work the [Department of Molecular Biosciences](https://molecularbiosci.utexas.edu/) at [The University of Texas at Austin](https://www.utexas.edu/).

Our support came from NIH grant 1R01 AI148419. Principle investigator C.O.W. was also supported by the Blumberg Centennial Professorship in Molecular Evolution and the Reeder Centennial Fellowship in Systematic and Evolutionary Biology at The University of Texas at Austin.
Computational analyses were performed using the Biomedical Research Computing Facility at UT Austin, Center for Biomedical Research Support. RRID: SCR_021979. The authors thank [Luiz Vieira](https://github.com/ziul-bio) for support and discussions on the topic of language model pretraining and finetuning.

## Table of Contents
- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Finetuning](#finetuning)
- [Models](#models)
- [Tokenizer](#tokenizer)
- [Datasets](#datasets)
- [Contributing](#contributing)
- [License](#license)

## Introduction

PeptideCLM-23M is a peptide-specific pretrained chemical language model published in [future publication]. 
The base model was pretrained with masked language modeling on 23 million molecules, divided approximately into two halves: 12M small molecules and 11M peptides.
Finetuned models are trained to predict membrane penetration for cyclic peptides from CycPeptMPDB.

This repository contains some example code for loading pre-trained weights, which include a Jupyter notebook example of model assessment and a python script for finetuning the pretrained model. 

## Getting Started
### Installation
To install the required libraries (all necessary dependencies not listed), run:
```
pip install torch transformers datasets SmilesPE pandas
```
Specific installation instructions can be found on the respective github/PyPi/etc.

### Usage
To use the pre-trained models, you can load them as follows:

```
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained('aaronfeller/model_name') 
```
Replace `'model_name'` with the desired model from the table below.

### Finetuning
Refer to the `example_training_script.py` script in this repository for an example of how to finetune the pretrained model on your dataset. You can run the script with: 
```
python example_finetuning_script.py
```

## Models
Required libraries for loading this model include the Transformers library with its dependencies (PyTorch, etc.) and the tokenizer library SmilesPE along with its dependencies.


| Model name              | Training dataset                                          | Description                                                                                                               |
|-----------------------------|--------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| `PeptideCLM-23M-all`         | PubChem + SureChEMBL + SmProt + RandPept | Fully pretrained PeptideCLM which performed best in finetuning on the downstream task of predicting peptide membrane penetration. |
| `PeptideCLM-11M-pep`         | SmProt + RandPept                        | PeptideCLM pretrained on the peptide portion of pretraining data. |
| `PeptideCLM-12M-smol`        | PubChem + SureChEMBL                     | PeptideCLM pretrained on the small molecule portion of pretraining data. |


All models hosted on huggingface can be loaded from [huggingface.co/aaronfeller](https://huggingface.co/aaronfeller).


## Tokenizer

A custom tokenizer was both inspired by and built using [Smiles Pair Encoding](https://github.com/XinhaoLi74/SmilesPE).

I attempted to port my custom tokenizer to HuggingFace, but was unable to. If anyone can sort this out, I'm happy to accept a pull request either here or to HuggingFace. For now, you can import the tokenizer as shown in either examples contained in this directory.

## Datasets

The pretraining dataset for the large model (23M) can be downloaded here: https://zenodo.org/records/15042141. 

**UPDATE** -- This record has two versions, the v1.1 has been corrected and should be used for any pretraining tasks. The first version had an issue with cyclization and the generated peptides did not have correct numbering for ring closure. All molecules should now be convertible to mol files with RDKit.

The clustered dataset of cyclic peptides analyzed with PAMPA from CycPeptMPDB is contained in the directory `clustered_data` with SMILES, permeability (PAMPA), and cluster number.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.

## License
The author(s) are protected under the MIT License - see the LICENSE file for details.

