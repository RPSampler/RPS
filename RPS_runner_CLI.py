

import time
import argparse
from RPS.RPS import RPS, Classifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, Perceptron
from sklearn.naive_bayes import MultinomialNB
import copy
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def run_all(model, streamdata, sample_size, dampingFactor, utilityMeasure, maxNorm, alpha, patternLanguage, batchsize, labled, weightedItems, predict_duration, learning_duration,classification_task):
    model_algo = copy.deepcopy(model)
    model_classifier =  Classifier(patternLanguage, model_algo)
    
    sampler = RPS(
         sample_size=sample_size,
         dampingFactor=dampingFactor, 
         utilityMeasure=utilityMeasure, 
         maxNorm=maxNorm,
         alpha=alpha,
         patternLanguage=patternLanguage,
         batchsize=batchsize,
         labled=labled,
         weightedItems=weightedItems,
         classification=("Y" == classification_task.upper()),
         predict_duration=predict_duration,
         learning_duration=learning_duration,
         model_classifier=model_classifier
    )
    
    return sampler.Sampler(streamdata) 

def execute(params):
    model_name, streamdata, sample_size, dampingFactor, batchsize, predict_duration, learning_duration, utilityMeasure, maxNorm, alphaDecay, patternLanguage, classification_task = params
    models = {
        'MultinomialNB': MultinomialNB(alpha=0.0001),
        'Perceptron': Perceptron(max_iter=10000, tol=1e-3),
        'PassiveAggressiveClassifier': PassiveAggressiveClassifier(max_iter=1000, tol=1e-3, loss='hinge'),
        'MLPClassifier': MLPClassifier(hidden_layer_sizes=(500,100,), activation='identity', solver='adam', max_iter=1000, warm_start=False, random_state=42),
        'SGDClassifier': SGDClassifier(loss='hinge', penalty='l1', alpha=0.0001, max_iter=1000, tol=1e-3)
    }

    model = models[model_name]
    labled = ("Y" == classification_task.upper())
    weightedItems = (utilityMeasure in ["HUI", "HAUI"])

    #logging.info(f"Processing file: {streamdata} with model: {model_name}, sample_size: {sample_size}, dampingFactor: {dampingFactor}, batchsize: {batchsize}, predict_duration: {predict_duration}, learning_duration: {learning_duration}")
    start_time = time.time()
    _, _ = run_all(
        model, streamdata, sample_size, dampingFactor, 
        utilityMeasure, maxNorm, alphaDecay, patternLanguage, 
        batchsize, labled, weightedItems, predict_duration, learning_duration, classification_task)
    elapsed_time = time.time() - start_time
    
    info = f"Global execution time: {elapsed_time} seconds"
    logging.info(info)

def main(args):
    models = ['MultinomialNB', 'Perceptron', 'PassiveAggressiveClassifier', 'MLPClassifier', 'SGDClassifier']
    
    welcome_message = f"""
    Welcome to the RPS Model Runner!
    
    Available Models:
        {', '.join(models)}
    Available Utility Measures:
        freq, area, decay, HUI, HAUI 
    Usable languages:
        * Sequence (can be combined with freq, area or decay) 
        * Itemset (can be combined any of them)
    
    Default Parameters:
    - Sample Size: {args.sample_size}
    - Damping Factor (ε): {args.dampingFactor}
    - Batch Size: {args.batchsize}
    - Learning Duration: {args.learning_duration}
    - Predict Duration: {args.predict_duration}
    - Utility Measure: {args.utilityMeasure}
    - Maximal Norm Constraint: {args.maxNorm}
    - Exponential decay (α):{args.alphaDecay}
    - Running model: {args.model_name}
    - Pattern language: {args.patternLanguage}
    - Classification task: {args.classification_task}
    - Stream file name: {args.data_dir}
    """
    print(welcome_message)
        
    model_name = args.model_name
    data_dir = args.data_dir
    sample_size = args.sample_size
    dampingFactor = args.dampingFactor
    batchsize = args.batchsize
    learning_duration = args.learning_duration
    predict_duration = args.predict_duration
    utilityMeasure = args.utilityMeasure
    maxNorm = args.maxNorm
    alphaDecay = args.alphaDecay
    patternLanguage = args.patternLanguage
    classification_task = args.classification_task
    
    new_param_values = input("Do you need to change parameters value? (Y/N): ")
    if new_param_values.upper() == "Y":
        model_name = input(f'Name of the classifier model to run [default {model_name}]: ') or model_name
        data_dir = input(f'Directory where the datasets are stored [default {data_dir}]: ') or data_dir
        sample_size = input(f'Size of the reservoir to use [default {sample_size}]: ') or sample_size
        dampingFactor = input(f'Damping factor for the model [default {dampingFactor}]: ') or dampingFactor
        batchsize = input(f'Batch size [default {batchsize}]: ') or batchsize
        learning_duration = input(f'Duration of the learning phase [default {learning_duration}]: ') or learning_duration
        predict_duration = input(f'Duration of the prediction phase [default {predict_duration}]: ') or predict_duration
        utilityMeasure = input(f'Utility measure [default {utilityMeasure}]: ') or utilityMeasure
        maxNorm = input(f'Maximal norm constraint [default {maxNorm}]: ') or maxNorm
        alphaDecay = input(f'Exponential decay value [default {alphaDecay}]: ') or alphaDecay
        patternLanguage = input(f'Pattern language [default {patternLanguage}]: ') or patternLanguage
        classification_task = input(f'Classification task ? (Y/N) [default {classification_task}]: ') or classification_task
    
    sample_size = int(sample_size)
    dampingFactor = float(dampingFactor)
    batchsize = int(batchsize)
    learning_duration = int(learning_duration)
    predict_duration = int(predict_duration)
    maxNorm = int(maxNorm)
    alphaDecay = float(alphaDecay)

    param_combination = (
        model_name, data_dir, 
        sample_size, dampingFactor, 
        batchsize, predict_duration, 
        learning_duration, utilityMeasure, maxNorm,
        alphaDecay, patternLanguage, classification_task
    )
    execute(param_combination)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run RPS models with different configurations.',
        usage='RPS_runner.py --model_name MODEL_NAME --data_dir DATA_DIR [--sample_size SAMPLE_SIZE] '
              '[--dampingFactor DAMPINGFACTOR] [--batchsize BATCHSIZE] '
              '[--learning_duration LEARNING_DURATION] [--predict_duration PREDICT_DURATION]'
    )
    parser.add_argument('--model_name', type=str, default="MultinomialNB", help='Name of the classifier model to run')
    parser.add_argument('--data_dir', type=str, default="Benchmark/Sequence/Books.num", help='Directory where the datasets are stored')
    parser.add_argument('--sample_size', type=int, default=10000, help='Size of the reservoir to use')
    parser.add_argument('--dampingFactor', type=float, default=0.1, help='Damping factor for the model')
    parser.add_argument('--batchsize', type=int, default=1000, help='Batch size')
    parser.add_argument('--learning_duration', type=int, default=3, help='Duration of the learning phase')
    parser.add_argument('--predict_duration', type=int, default=20, help='Duration of the prediction phase')
    parser.add_argument('--utilityMeasure', type=str, default='decay', help='Utility measure')
    parser.add_argument('--maxNorm', type=int, default=10, help='Maximal norm constraint')
    parser.add_argument('--alphaDecay', type=float, default=0.001, help='Exponential decay')
    parser.add_argument('--patternLanguage', type=str, default='Sequence', help='Pattern language')
    parser.add_argument('--classification_task', type=str, default="Y", help='Use for classification task')
    
    args = parser.parse_args()        
    main(args)


