# RPS: A Generic Reservoir Patterns Sampler

## A. Overview

Efficient learning from streaming data is important in data analysis, given the continuous evolution of data streams. Despite significant strides in stream pattern mining, challenges persist, particularly with managing complex data streams such as sequential and weighted itemsets. Reservoir sampling, a fundamental method for randomly selecting fixed-size samples from data streams, has yet to be fully explored for these complex patterns.

This repository introduces an innovative approach using weighted reservoir sampling to facilitate direct pattern sampling from streaming batch data, ensuring scalability and efficiency. Our generic algorithm addresses temporal biases and supports various pattern types, including sequential, weighted, and unweighted itemsets.

## B. Key Features

- Supports multiple classifier models: `MultinomialNB`, `Perceptron`, `PassiveAggressiveClassifier`, `MLPClassifier`, `SGDClassifier`.
- Handles complex data streams with `Sequence` and `Itemset` pattern languages.
- Diverse utility measures `freq`, `area`, `decay`, `HUI`, `HAUI`. 
- Configurable parameters including sample size (reservoir size), damping factor, batch size, learning and prediction durations, utility measures, and more.
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
The following figure shows the behavior of ${\bf RPS}$ on different sequential databases as the reservoir size increases with different batch sizes. The experiments are repeated 5 times, and the standard deviations are tiny and not visible in the figure. While the reservoir size has a slight impact on the execution time, the batch size affects only the execution time per batch, not the overall execution time of the entire dataset. This is because ${\bf RPS}$ weights each instance independently of the batch it belongs to.
<figcaption>
  
![image](https://github.com/RPSampler/RPS/assets/172807587/035a237c-aa53-4beb-ab8f-03c7cb63ec32)

*Fig 1. Impact of the reservoir and batch size on the speed*
</figcaption> 


### D.3. Experimental results for unweighted Itemset and Weighted-Itemset
These benchmarks contain real-world databases sourced from the SPMF repository (https://www.philippe-fournier-viger.com/spmf/index.php?link=datasets.php). *Tab 1* contains unweighted itemset datasets, while *Tab 2* contains weighted itemsets for high utility itemsets (HUI) and high average-utility itemsets (HAUI) discovery.


<tabcaption> *Tab 1: Our benchmark on unweighted Itemsets$
| Database | $\|{\cal D}\|$ | $\|I\|$ | $\|\|γ\|\|_{max}$ | $\|\|γ\|\|_{avg}$ |
|----------|----------------|---------|-------------------|-------------------|
| ORetail  | 541,909        | 2,603   | 8                 | 4.37              |
| Kddcup99 | 1,000,000      | 135     | 16                | 16                |
| PowerC   | 1,040,000      | 140     | 7                 | 7                 |
| Susy     | 5,000,000      | 190     | 19                | 19                |
</tabcaption>


<tabcaption> *Tab 2:benchmark on weighted itemsets (For HUI and HAUI)*
| Database           | $\|{\cal D}\|$     | $\|I\|$   | $\|\|γ\|\|_{max}$ | $\|\|γ\|\|_{avg}$ |
|--------------------|-----------|--------|----------------------|----------------------|
| ECommerce          | 14,975    | 3,468  | 29                   | 11.71                |
| Fruithut         | 181,970   | 1,265  | 36                   | 3.59                 |
| Chainstore       | 1,112,949 | 46,086 | 168                  | 7.23                 |
| ChicagoC      | 2,662,309 | 35     | 13                   | 1.79                 |
</tabcaption>



**a) Performance analysis of RPS algorithm across itemset pattern language with diverse Database characteristics and parameter settings**

The ${\bf RPS}$ algorithm demonstrates in Table 3 efficient performance across a spectrum of database characteristics, as illustrated by the findings from Tables 1 and 2. In unweighted databases such as ORetail and Kddcup99 (from Table 1), characterized by moderate to large transaction and item counts ($\|{\cal D}\|$) and ($\|I\|$), ${\bf RPS}$ operates with commendably low execution times. This efficiency extends seamlessly to larger datasets, highlighting ${\bf RPS}$ ability to maintain rapid processing speeds even when handling extensive data volumes.

Conversely, in weighted databases like ECommerce and Fruithut (from Table 2), which involve higher transaction volumes and intricate itemset calculations, ${\bf RPS}$ exhibits slightly longer execution times. Nevertheless, ${\bf RPS}$ consistently performs well, delivering manageable processing times across various parameter configurations, including scenarios with larger reservoir sizes and higher damping factors ($ε$).

In general, the performance analysis underscores ${\bf RPS}$ robust capability to handle both unweighted and weighted databases effectively. This versatility makes ${\bf RPS}$ a valuable approach for scalable stream data mining applications, emphasizing its reliability in maintaining efficient processing speeds while accommodating diverse database complexities.

<tabcaption> *Tab 3: Average execution time per batch (in seconds) with different values of the damping factor* ($ε \in \\{ 0.0, 0.1, 0.5 \\}$), *the batch size* ($=1000$), *the reservoir size* ($k \in \\{ 1000, 2000, 3000 \\}$), *and maximal norm constraint* $M=10$.

![image](https://github.com/RPSampler/RPS/assets/172807587/f385a416-c8e0-44b2-802c-9f1eb5cf759a)


**b) Speed comparison between ${\bf 2-Step}$ *(Boley et al.,KDD'11)*, ${\bf ResPat}$ *(Giacometti \& Soulet, ECML-PKDD'22) and* ${\bf RPS}$ on unweighted itemsets databases**

*Tab 3* contains execution time comparisons between ${\bf 2-Step}$ *(Boley et al.,KDD'11)*, ${\bf ResPat}$ (Giacometti \& Soulet, ECML-PKDD'22) and our approach, ${\bf RPS}$. The experiments were repeated 5 times with different damping factors ($\epsilon = \\{0.0, 0.1, 0.5 \\}$), a sample size of $k=10,000$ without norm constraint (i.e., $M=\infty$), and the standard deviations are reported. We set a maximal execution time of $\textbf{1 hour (3600 seconds)}$, and the symbol $(-)$ indicates that the approach exceeded the time limit (1 hour) for the corresponding dataset. We set "*oom*" for out of memory when the data cannot be processed by the corresponding approach.

<tabcaption> *Tab 3: Experimental results on the execution times without length constraint for* ${\bf ResPat}$ *and* ${\bf RPS}$
| Database  | ${\bf 2-Step}$ | ${\bf ResPat}(ε=0)$ | ${\bf ResPat}(ε=0.1)$ | ${\bf ResPat}(ε=0.5)$ | ${\bf RPS}(ε=0)$ | ${\bf RPS}(ε=0.1)$ | ${\bf RPS}(ε=0.5)$ |
|-----------|-------------------|------------------------|--------------------------|-------------------------|---------------------|-----------------------|-----------------------|
| ORetail   | $1.03 \pm 0.04$   | $190.30 \pm 0.69$      | $2,650.79 \pm 26.85$      | $255.26 \pm 0.11$       | $0.75 \pm 0.01$     | $4.24 \pm 0.03$       | $6.33 \pm 0.14$       |
| Kddcup99  | $3.08 \pm 0.16$   | $412.28 \pm 0.56$      | $-$                      | $-$                     | $1.53 \pm 0.02$     | $8.54 \pm 0.17$       | $14.05 \pm 0.31$      |
| PowerC    | $2.41 \pm 0.22$   | $248.52 \pm 0.94$      | $-$                      | $2,233.44 \pm 14.94$     | $1.35 \pm 0.01$     | $8.06 \pm 0.05$       | $12.91 \pm 0.11$      |
| Susy      | $oom$             | $513.37 \pm 2.16$      | $-$                      | $-$                     | $8.53 \pm 0.18$     | $45.25 \pm 0.48$      | $77.86 \pm 6.08$      |
</tabcaption>

Upon examining the results presented in Table 3, it is evident that the ${\bf RPS}$ approach demonstrates superior performance and versatility compared to both ${\bf 2-Step}$ and ${\bf ResPat}$ methods across all damping factors.

For the damping factor of 0 (ε=0), ${\bf RPS}$ consistently outperforms ${\bf ResPat}$ in terms of execution times for all datasets. Specifically, ${\bf RPS}$ processes the ORetail, Kddcup99, PowerC, and Susy datasets in $0.75 \pm 0.01$, $1.53 \pm 0.02$, $1.35 \pm 0.01$, and $8.53 \pm 0.18$ seconds, respectively. In contrast, ${\bf ResPat}$ takes $190.30 \pm 0.69$, $412.28 \pm 0.56$, $248.52 \pm 0.94$, and $513.37 \pm 2.16$ seconds, respectively, for the same datasets. Moreover, ${\bf RPS}$ successfully handles the Susy dataset, where ${\bf 2-Step}$ fails due to an out-of-memory error, highlighting ${\bf RPS}$'s ability to manage larger datasets efficiently.

When the damping factor is set to 0.1 (ε=0.1), ${\bf RPS}$ continues to exhibit shorter execution times compared to ${\bf ResPat}$ for all datasets. In this case, ${\bf RPS}$ completes the processing of the ORetail, Kddcup99, PowerC, and Susy datasets in $4.24 \pm 0.03$, $8.54 \pm 0.17$, $8.06 \pm 0.05$, and $45.25 \pm 0.48$ seconds, respectively. Meanwhile, ${\bf ResPat}$ exceeds the 1-hour time limit for the Kddcup99, PowerC and Susy datasets and takes $2,650.79 \pm 26.85$ seconds for the ORetail dataset. Again, ${\bf RPS}$ proves to be more efficient in handling larger datasets that ${\bf ResPat}$ struggles with.

Lastly, for the damping factor of 0.5 (ε=0.5), ${\bf RPS}$ maintains its advantage over ${\bf ResPat}$ in terms of execution times. ${\bf RPS}$ processes the ORetail, Kddcup99, PowerC, and Susy datasets in $6.33 \pm 0.14$, $14.05 \pm 0.31$, $12.91 \pm 0.11$, and $77.86 \pm 6.08$ seconds, respectively. ${\bf ResPat}$, on the other hand, exceeds the 1-hour time limit for the Kddcup99 and Susy datasets and takes $255.26 \pm 0.11$ and $2,233.44 \pm 14.94$ seconds for the ORetail and PowerC datasets, respectively.

In summary, the ${\bf RPS}$ approach consistently outperforms ${\bf ResPat}$ across all damping factors and datasets, demonstrating its efficiency and scalability. Furthermore, ${\bf RPS}$ effectively handles larger datasets, such as Susy, that ${\bf 2-Step}$ cannot process due to memory constraints. These findings underscore the robustness and adaptability of the ${\bf RPS}$ approach for various datasets and damping factor settings.





