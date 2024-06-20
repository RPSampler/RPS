# RPS: A Generic Reservoir Patterns Sampler

## A. Overview

Efficient learning from streaming data is important in data analysis, given the continuous evolution of data streams. Despite significant strides in stream pattern mining, challenges persist, particularly with managing complex data streams such as sequential and weighted itemsets. Reservoir sampling, a fundamental method for randomly selecting fixed-size samples from data streams, has yet to be fully explored for these complex patterns.

This repository introduces an innovative approach using weighted reservoir sampling to facilitate direct pattern sampling from streaming batch data, ensuring scalability and efficiency. Our generic algorithm addresses temporal biases and supports various pattern types, including sequential, weighted, and unweighted itemsets.

## B. Key Features

- Supports multiple classifier models: `MultinomialNB`, `Perceptron`, `PassiveAggressiveClassifier`, `MLPClassifier`, `SGDClassifier`.
- Handles complex data streams with `Sequence` and `Itemset` pattern languages.
- Diverse utility measures `freq`, `area`, `decay`, `HUI`, `HAUI`. 
- Configurable parameters including sample size, damping factor, batch size, learning and prediction durations, utility measures, and more.
- GUI for easy configuration and execution.
- CLI for advanced users and scripting.

## C. Framework implementation

### C.1. Requirements

Before running the code, ensure you have the following Python dependencies installed:
- numpy
- scipy
- scikit-learn
- tkinter
pip install -r requirements.txt

git clone https://github.com/RPSampler/RPS.git

cd RPS

pip install -r requirements.txt

### C.2. Graphical User Interface (GUI)
python3 RPS_runner_GUI.py
### C.3. Command-Line Interface (CLI)
python3 RPS_runner_CLI.py --model_name MODEL_NAME --data_dir DATA_DIR [--sample_size SAMPLE_SIZE] [--dampingFactor DAMPINGFACTOR] [--batchsize BATCHSIZE] [--learning_duration LEARNING_DURATION] [--predict_duration PREDICT_DURATION] [--utilityMeasure UTILITYMEASURE] [--maxNorm MAXNORM] [--alphaDecay ALPHADECAY] [--patternLanguage PATTERNLANGUAGE] [--classification_task CLASSIFICATIONTASK] [--labeled_data LABELED_DATA]

*Exemple:* python3 RPS_runner_CLI.py --model_name MultinomialNB --data_dir Benchmark/Sequence/Books.num --sample_size 10000 --dampingFactor 0.1 --batchsize 1000 --learning_duration 3 --predict_duration 20 --utilityMeasure area --maxNorm 5 --alphaDecay 0.001 --patternLanguage Sequence --classification_task N --labeled_data Y

#### NB: It is also possible to run RPS_runner_CLI.py without arguments and change the default values step by step if needed.

### C.4. Parameters

- `model_name`: Name of the classifier model to run (default: `MultinomialNB`).
- `data_dir`: Directory where the datasets are stored (default: `Benchmark/Sequence/Books.num`).
- `sample_size`: Size of the reservoir to use (default: `5000`).
- `dampingFactor`: Damping factor for the model (default: `0.1`).
- `batchsize`: Batch size (default: `1000`).
- `learning_duration`: Duration of the learning phase (default: `3`).
- `predict_duration`: Duration of the prediction phase (default: `20`).
- `utilityMeasure`: Utility measure (`freq`, `area`, `decay`, `HUI`, `HAUI`) (default: `area`).
- `maxNorm`: Maximal norm constraint (default: `5`).
- `alphaDecay`: Exponential decay (default: `0.001`).
- `patternLanguage`: Pattern language (`Sequence` or `Itemset`) (default: `Sequence`).
- `classification_task`: Whether to use for classification task (`Y` or `N`) (default: `N`).
- `labeled_data `: Whether the data is labeled (`Y` or `N`) (default: `Y`).

## D. Additional results

### D.1. Theoretical results

![image](https://github.com/RPSampler/RPS/assets/172807587/63330266-8bbf-4d30-843c-71445672efa6)

### D.2. Additional experimental results for the sequential pattern language
The following figure shows the behavior of RPS on different sequential databases as the reservoir size increases with different batch sizes. The experiments are repeated 5 times, and the standard deviations are tiny and not visible in the figure. While the reservoir size has a slight impact on the execution time, the batch size affects only the execution time per batch, not the overall execution time of the entire dataset. This is because RPS weights each instance independently of the batch it belongs to.
<figcaption>
  
![image](https://github.com/RPSampler/RPS/assets/172807587/035a237c-aa53-4beb-ab8f-03c7cb63ec32)

Fig 1. Impact of the reservoir and batch size on the speed
</figcaption>


### D.3. Experimental results for unweighted Itemset and Weighted-Itemset

<tabcaption> Tab 1: Our benchmark on Itemsets
| Database           | \|D\|     | \|I\|   | \|\|γ\|\|<sub>max</sub>| \|\|γ\|\|<sub>avg</sub>  |
|--------------------|-----------|--------|----------------------|----------------------|
| ORetail       | 541,909   | 2,603  | 8                    | 4.37                 |
| Kddcup99           | 1,000,000 | 135    | 16                   | 16                   |
| PowerC             | 1,040,000 | 140    | 7                    | 7                    |
| Susy               | 5,000,000 | 190    | 19                   | 19                   |
</tabcaption>

<tabcaption> Tab 2:benchmark on Weighted itemsets (For HUI and HAUI)
| Database           | \|D\|     | \|I\|   | \|\|γ\|\|<sub>max</sub>| \|\|γ\|\|<sub>avg</sub> |
|--------------------|-----------|--------|----------------------|----------------------|
| ECommerce          | 14,975    | 3,468  | 29                   | 11.71                |
| Fruithut         | 181,970   | 1,265  | 36                   | 3.59                 |
| Chainstore       | 1,112,949 | 46,086 | 168                  | 7.23                 |
| ChicagoC      | 2,662,309 | 35     | 13                   | 1.79                 |
</tabcaption>



**Performance analysis of RPS algorithm across itemset pattern language with diverse Database characteristics and parameter settings**

The RPS algorithm demonstrates in Table 3 efficient performance across a spectrum of database characteristics, as illustrated by the findings from Tables 1 and 2. In unweighted databases such as ORetail and Kddcup99 (from Table 1), characterized by moderate to large transaction and item counts (\(|D|\) and \(|I|\)), RPS operates with commendably low execution times. This efficiency extends seamlessly to larger datasets, highlighting RPS ability to maintain rapid processing speeds even when handling extensive data volumes.

Conversely, in weighted databases like ECommerce and Fruithut (from Table 2), which involve higher transaction volumes and intricate itemset calculations, RPS exhibits slightly longer execution times. Nevertheless, RPS consistently performs well, delivering manageable processing times across various parameter configurations, including scenarios with larger reservoir sizes and higher damping factors (ε).

In general, the performance analysis underscores RPS robust capability to handle both unweighted and weighted databases effectively. This versatility makes RPS a valuable approach for scalable stream data mining applications, emphasizing its reliability in maintaining efficient processing speeds while accommodating diverse database complexities.

<tabcaption> Tab 2:

![image](https://github.com/RPSampler/RPS/assets/172807587/540bcde5-dfb9-428b-9248-70bf0d7e6a12)

</tabcaption>



