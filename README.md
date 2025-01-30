# Enhancing Performance of Explainable AI Modelswith Constrained Concept Refinement

This repository contains code to accompany the submission of [*Enhancing Performance of Explainable AI Modelswith Constrained Concept Refinement*].

## Synthetic Experiments
The synthetic experiments introudced in Appendix D.1 validates the results proved in Theorem 3.3 and 3.4. These experiments are contained in synthetic_test.ipynb.

## Image Classification Experiments
This part is built on the repository of the paper Information Maximization Perspective of Orthogonal Matching Pursuit with Applications to Explainable AI, https://github.com/r-zip/ip-omp.git.

### Setup, Datasets and Preprocessing
To onboard the project, one should:
1. Download from https://github.com/r-zip/ip-omp.git and add clip_embedding_generation.py to its ip_omp folder.
2. follow the setup and data preprocessing instructions entailed in the readme file of https://github.com/r-zip/ip-omp.git.
3. Specifically because of their size, one need to download ImageNet and Places365 by themselves, add them to the designated folder and run ip_omp.preprocess in order to run the later experiments.

### Getting the CLIP concept/training/testing embedding.

The code reads training and test set for the input dataset and saves the CLIP embeddings for each image into a .npy file in the saved_files directory, alongside with the embeddings for the concepts, referred to as dictionary in the code.

Usage: python -m ip_omp.clip_embedding_generation -dataset {dataset name}

dataset name is a value from the set {"imagenet", "places365", "cub", "cifar10", "cifar100"}.

### Run Constrined Concept Refinement(CCR).

The experiments are contained in interpretable_image_classification.ipynb.

## License and Citation
This repository is MIT-licensed. We will complete this part after the reviewing process.
