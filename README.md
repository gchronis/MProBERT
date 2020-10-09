# Overview

This repository contains scripts for generating multi-prototype embeddings from BERT. They collect tokens of target words from the BNC, generate BERT representations for those tokens, cluster them, and analyze the resultant models on similarity and relatedness estimation, as well as concreteness estimation.


## Prerequisites

Python version
- 3.7.4

Package Dependencies
- torch
- pytorch_pretrained_bert
- nltk

Scripts require the following datasets in the `data` directory:
- BLESS
- MEN
- SimLex-999
- verbsim
- wordsim353_sim_rel
- simverb3500
- wordsim353

## Instructions

Scripts must be run from the `./scripts` subdirectory. Each script performs one of the steps for each dataset. To run a script for a single dataset, the code must be edited to comment out the other datasets.

In order to construct multiprototype embeddings, run these scripts in sequence:

- collect_tokens.py
- calculate_clusters.py

In order to evaluate mprobert emebddings against datasets, run
- layer_and_cluster_analysis.py 	#  correlates models against similarity and relatedness ratings for 7 datasets
- multicluster_layer_analysis.py    #  same as above, for unioned models
- abstractness_simlex_analysis.py
- abstractness_simlex_analysis_pos.py


To visualize results, run
- heatmap.py
- visualize_abstractness_simlex_analysis.py
- TSNE_similarity_visualization_reusable.ipynb (with jupyter notebook)