# HRDoc

This is the official PyTorch implementation of the paper: "HRDoc: Dataset and Baseline Method Toward Hierarchical Reconstruction of Document Structures".

<img src="./figs/model.jpg" width=600 alt="framework"/>

We build HRDoc with line-level annotations and cross-page relations that support both NLP and CV research. HRDoc dataset aims to recover the semantic structure of the PDF document, which can be divided into three subtasks, includ- ing semantic unit classification, parent finding, and relation classification.

<img src="./figs/task_intro.jpg" width=600 alt="framework"/>

## Dataset Introduction

HRDoc contains 2,500 documents with more than 2 million semantic units. The following figure provides the statistics of semantic unit distribution over the train and test set of the HRDoc datasets.

<img src="./figs/dataset_statistic.jpg" width=600 alt="framework"/>

Here we present some samples in the HRDoc dataset.

<img src="./figs/dataset_sample.jpg" width=600 alt="framework"/>

## Release Issues

We have released scripts related to the data generation, rule-based system and including 4 parts: 
1. The scripts used to get a colorized document. See `utils/pdf_colorization.py`
2. PDF parser systems used to extract the text lines, equation, table, and figure areas. See `utils/extract_pdf_hrdh.py` and `utils/extract_pdf_hrdh.py`
3. The human-designed rule system as a new baseline. See `utils/relation_recover.py`
4. An end2end multi-modal system for reconstruction task. See `end2end_system` folder

The HRDoc dataset and the model weight and inference code of DSPS system will be made available soon.