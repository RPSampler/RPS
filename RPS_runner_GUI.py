__author__ = ""

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import time
import copy
import logging
from RPS.RPS import RPS, Classifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, Perceptron
from sklearn.naive_bayes import MultinomialNB

# Set up logging
logging.basicConfig(level=logging.INFO)

def run_all(model, streamdata, sample_size, dampingFactor, utilityMeasure, maxNorm, alpha, patternLanguage, batchsize, labled, weightedItems, predict_duration, learning_duration, classification_task):
    model_algo = copy.deepcopy(model)
    model_classifier = Classifier(patternLanguage, model_algo)
    
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

def is_combination_valid(utilityMeasure, patternLanguage):
    valid_combinations = {
        'Sequence': ['freq', 'area', 'decay'],
        'Itemset': ['freq', 'area', 'decay', 'HUI', 'HAUI']
    }
    
    if patternLanguage not in valid_combinations:
        return False
    
    allowed_utilities = valid_combinations[patternLanguage]
    return utilityMeasure in allowed_utilities

def is_directory_compatible_with_pattern(data_directory, patternLanguage, utilityMeasure):
    return ("benchmark/"+patternLanguage.lower() in data_directory.lower()) or (utilityMeasure.lower() in data_directory.lower())

def execute(params, output_text, reservoir_text):
    model_name, streamdata, sample_size, dampingFactor, batchsize, predict_duration, learning_duration, utilityMeasure, maxNorm, alphaDecay, patternLanguage, classification_task = params
    
    # Check if the combination of utility measure and pattern language is valid
    if not is_combination_valid(utilityMeasure, patternLanguage):
        messagebox.showerror("Invalid Combination", f"The combination of Utility Measure '{utilityMeasure}' and Pattern Language '{patternLanguage}' is not valid.")
        return

    # Check if the pattern language is part of the directory data name
    if not is_directory_compatible_with_pattern(streamdata, patternLanguage, utilityMeasure):
        messagebox.showerror("Incompatible Data Directory", f"The data directory '{streamdata}' is not compatible with the pattern language '{patternLanguage}'.")
        return
    
    models = {
        'MultinomialNB': MultinomialNB(alpha=0.0001),
        'Perceptron': Perceptron(max_iter=10000, tol=1e-3),
        'PassiveAggressiveClassifier': PassiveAggressiveClassifier(max_iter=1000, tol=1e-3, loss='hinge'),
        'MLPClassifier': MLPClassifier(hidden_layer_sizes=(500,100,), activation='identity', solver='adam', max_iter=1000, warm_start=False, random_state=42),
        'SGDClassifier': SGDClassifier(loss='hinge', penalty='l1', alpha=0.0001, max_iter=1000, tol=1e-3)
    }

    model = models[model_name]
    labled = ("Y" == classification_task.upper())
    weightedItems = (utilityMeasure == 'HUI' or utilityMeasure ==  'HAUI')

    start_time = time.time()
    
    try:
        reservoir, _ = run_all(
            model, streamdata, sample_size, dampingFactor, 
            utilityMeasure, maxNorm, alphaDecay, patternLanguage, 
            batchsize, labled, weightedItems, predict_duration, learning_duration, classification_task)
    except ValueError as e:
        messagebox.showerror("Error", str(e))
        return
    
    elapsed_time = time.time() - start_time
    
    info = f"Global execution time: {elapsed_time} seconds"
    logging.info(info)
    output_text.insert(tk.END, info + "\n")
    reservoir_content = "Reservoir Sample:\n"
    for patt in reservoir:
        reservoir_content += str(patt)+"\n"
        
    reservoir_text.insert(tk.END, f"{reservoir_content}")

def run_model(output_text, reservoir_text):
    # Clear previous output in the output_text widget
    output_text.delete(1.0, tk.END)
    
    params = (
        model_name_var.get(),
        data_dir_var.get(),
        int(sample_size_var.get()),
        float(dampingFactor_var.get()),
        int(batchsize_var.get()),
        int(predict_duration_var.get()),
        int(learning_duration_var.get()),
        utilityMeasure_var.get(),
        int(maxNorm_var.get()),
        float(alphaDecay_var.get()),
        patternLanguage_var.get(),
        classification_task_var.get()
    )
    execute(params, output_text, reservoir_text)

# Create the main window
root = tk.Tk()
root.title("RPS Model Runner")

# Define StringVars for the entries
model_name_var = tk.StringVar(value="MultinomialNB")
data_dir_var = tk.StringVar(value="Benchmark/Sequence/Books.num")
sample_size_var = tk.StringVar(value="5000")
dampingFactor_var = tk.StringVar(value="0.1")
batchsize_var = tk.StringVar(value="1000")
learning_duration_var = tk.StringVar(value="3")
predict_duration_var = tk.StringVar(value="20")
utilityMeasure_var = tk.StringVar(value="decay")
maxNorm_var = tk.StringVar(value="2")
alphaDecay_var = tk.StringVar(value="0.001")
patternLanguage_var = tk.StringVar(value="Sequence")
classification_task_var = tk.StringVar(value="Y")

# Define the options for the comboboxes
model_options = ['MultinomialNB', 'Perceptron', 'PassiveAggressiveClassifier', 'MLPClassifier', 'SGDClassifier']
utility_measure_options = ['area', 'decay', 'freq', 'HAUI', 'HUI']
pattern_language_options = ['Sequence', 'Itemset']

# Define labels and entries/comboboxes
labels_and_vars = [
    ("Model Name:", model_name_var, model_options),
    ("Data Directory:", data_dir_var, None),
    ("Sample Size:", sample_size_var, None),
    ("Damping Factor:", dampingFactor_var, None),
    ("Batch Size:", batchsize_var, None),
    ("Learning Duration:", learning_duration_var, None),
    ("Predict Duration:", predict_duration_var, None),
    ("Utility Measure:", utilityMeasure_var, utility_measure_options),
    ("Maximal Norm Constraint:", maxNorm_var, None),
    ("Exponential Decay:", alphaDecay_var, None),
    ("Pattern Language:", patternLanguage_var, pattern_language_options),
    ("Classification Task (Y/N):", classification_task_var, ['Y', 'N'])
]

for i, (label, var, options) in enumerate(labels_and_vars):
    ttk.Label(root, text=label).grid(row=i, column=0, padx=10, pady=5, sticky='w')
    if label == "Data Directory:":
        entry = ttk.Entry(root, textvariable=var, width=50)
        entry.grid(row=i, column=1, padx=10, pady=5, sticky='ew')
        ttk.Button(root, text="Browse", command=lambda v=var: browse_file(v)).grid(row=i, column=2, padx=5, pady=5)
    elif options:
        combobox = ttk.Combobox(root, textvariable=var, values=options)
        combobox.grid(row=i, column=1, padx=10, pady=5, sticky='ew')
        combobox.current(0)
    else:
        ttk.Entry(root, textvariable=var).grid(row=i, column=1, padx=10, pady=5, sticky='ew')

# Run button
run_button = ttk.Button(root, text="Run Model")
run_button.grid(row=len(labels_and_vars), column=0, columnspan=3, pady=10)

# Output text widget for elapsed time and reservoir sample
output_text = tk.Text(root, height=15, width=80)
output_text.grid(row=len(labels_and_vars)+1, column=0, columnspan=3, padx=10, pady=10, sticky='nsew')

# Create a scrollbar for the output text widget
scrollbar = ttk.Scrollbar(root, command=output_text.yview)
scrollbar.grid(row=len(labels_and_vars)+1, column=3, sticky='nsew')
output_text.config(yscrollcommand=scrollbar.set)

def browse_file(var):
    filename = filedialog.askopenfilename(initialdir="Benchmark/", title="Select File", filetypes=[("All Files", "*.*")])
    if filename:
        var.set(filename)

# Function to redirect the button to the run_model and clear the output_text widget
def run_model_and_clear_output():
    output_text.delete(1.0, tk.END)  # Clear previous output
    run_model(output_text, output_text)  # Directly use output_text for reservoir_text since they are combined

# Validate combination and run model
def validate_and_run_model():
    utilityMeasure = utilityMeasure_var.get()
    patternLanguage = patternLanguage_var.get()
    
    if not is_combination_valid(utilityMeasure, patternLanguage):
        messagebox.showerror("Invalid Combination", f"The combination of Utility Measure '{utilityMeasure}' and Pattern Language '{patternLanguage}' is not valid.")
    elif not is_directory_compatible_with_pattern(data_dir_var.get(), patternLanguage, utilityMeasure):
        messagebox.showerror("Incompatible Data Directory", f"The data directory '{data_dir_var.get()}' is not compatible with the pattern language '{patternLanguage}'.")
    else:
        run_model_and_clear_output()

run_button.config(command=validate_and_run_model)

root.mainloop()
