# HRDoc

This is the official PyTorch implementation of the paper: "HRDoc: Dataset and Baseline Method Toward Hierarchical Reconstruction of Document Structures".

<img src="./figs/model.jpg" alt="framework"/>

We build HRDoc with line-level annotations and cross-page relations that support both NLP and CV research. HRDoc dataset aims to recover the semantic structure of the PDF document, which can be divided into three subtasks, includ- ing semantic unit classification, parent finding, and relation classification.

<img src="./figs/task_intro.jpg" alt="framework"/>

## Dataset Introduction

HRDoc contains 2,500 documents with more than 2 million semantic units. The following figure provides the statistics of semantic unit distribution over the train and test set of the HRDoc datasets.

<img src="./figs/dataset_statistic.jpg" alt="framework"/>

Here we present some samples in the HRDoc dataset.

<img src="./figs/dataset_sample.jpg" alt="framework"/>

## Release Issues

All code related to train and test will be released in near future.