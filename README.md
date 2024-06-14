# RPS: A Generic Reservoir Patterns Sampler

## Overview

Efficient learning from streaming data is crucial in contemporary data analysis, given the continuous evolution of data streams. Despite significant strides in stream pattern mining, challenges persist, particularly with managing complex data streams such as sequential and weighted itemsets. Reservoir sampling, a fundamental method for randomly selecting fixed-size samples from data streams, has yet to be fully explored for these complex patterns.

This repository introduces an innovative approach using weighted reservoir sampling to facilitate direct pattern sampling from streaming batch data, ensuring scalability and efficiency. Our generic algorithm addresses temporal biases and supports various pattern types, including sequential, weighted, and unweighted itemsets.

## Key Features

- Supports multiple classifier models: `MultinomialNB`, `Perceptron`, `PassiveAggressiveClassifier`, `MLPClassifier`, `SGDClassifier`.
- Handles complex data streams with `Sequence` and `Itemset` pattern languages.
- Diverse utility measures `freq`, `area`, `decay`, `HUI`, `HAUI`. 
- Configurable parameters including sample size, damping factor, batch size, learning and prediction durations, utility measures, and more.
- GUI for easy configuration and execution.
- CLI for advanced users and scripting.

## Requirements

Before running the code, ensure you have the following Python dependencies installed:
- numpy
- scipy
- scikit-learn
- tkinter
pip install -r requirements.txt

git clone https://github.com/RPSampler/RPS.git

cd RPS

pip install -r requirements.txt

## Graphical User Interface (GUI)
python3 RPSampler_GUI.py

## Command-Line Interface (CLI)
python3 RPSampler_CLI.py --model_name MODEL_NAME --data_dir DATA_DIR [--sample_size SAMPLE_SIZE] [--dampingFactor DAMPINGFACTOR] [--batchsize BATCHSIZE] [--learning_duration LEARNING_DURATION] [--predict_duration PREDICT_DURATION] [--utilityMeasure UTILITYMEASURE] [--maxNorm MAXNORM] [--alphaDecay ALPHADECAY] [--patternLanguage PATTERNLANGUAGE] [--classification_task CLASSIFICATIONTASK]

*Exemple:* python3 RPS_runner.py --model_name MultinomialNB --data_dir Benchmark/Sequence/Books.num --sample_size 10000 --dampingFactor 0.1 --batchsize 1000 --learning_duration 3 --predict_duration 20 --utilityMeasure decay --maxNorm 10 --alphaDecay 0.001 --patternLanguage Sequence --classification_task Y

## Parameters

- `model_name`: Name of the classifier model to run (default: `MultinomialNB`).
- `data_dir`: Directory where the datasets are stored (default: `Benchmark/Sequence/Books.num`).
- `sample_size`: Size of the reservoir to use (default: `10000`).
- `dampingFactor`: Damping factor for the model (default: `0.1`).
- `batchsize`: Batch size (default: `1000`).
- `learning_duration`: Duration of the learning phase (default: `3`).
- `predict_duration`: Duration of the prediction phase (default: `20`).
- `utilityMeasure`: Utility measure (`freq`, `area`, `decay`, `HUI`, `HAUI`) (default: `decay`).
- `maxNorm`: Maximal norm constraint (default: `10`).
- `alphaDecay`: Exponential decay (default: `0.001`).
- `patternLanguage`: Pattern language (`Sequence` or `Itemset`) (default: `Sequence`).
- `classification_task`: Whether to use for classification task (`Y` or `N`) (default: `Y`).

# Additional results

## Theoretical results

![image](https://github.com/RPSampler/RPS/assets/172807587/63330266-8bbf-4d30-843c-71445672efa6)
