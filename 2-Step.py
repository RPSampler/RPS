import time
from random import random
from decimal import Decimal

def find(tab, i, j, x):
    """
    Binary search to find the index where x should be inserted into tab.
    
    Args:
    - tab (list): List of cumulative weights.
    - i (int): Start index of the search.
    - j (int): End index of the search.
    - x (Decimal): Randomly generated value to find.

    Returns:
    - int: Index in tab where x should be inserted.
    """
    m = int((i + j) / 2)
    if m == 0 or (tab[m - 1] < x and x <= tab[m]):
        return m
    if tab[m] < x:
        return find(tab, m + 1, j, x)
    return find(tab, i, m, x)

def TwoStep(dataset, sample_size):
    """
    Function to perform the TwoStep sampling method.

    Args:
    - dataset (str): Path to the dataset file.
    - sample_size (int): Number of samples to generate.

    Returns:
    - list: List of sampled patterns (sets of items).
    """
    tab_data = []  # List to store sets of items from the dataset
    w_data = []    # List to store cumulative weights
    z = Decimal(0.)

    # Reading the dataset
    with open(dataset, 'r') as base:
        line = base.readline()
        while line:
            instance = line.split()
            w = 2 ** len(instance)  # Weight calculation
            tab_data.append(set(instance))  # Storing each instance as a set
            z += Decimal(w)  # Accumulating the total weight
            w_data.append(z)  # Storing cumulative weights
            line = base.readline()

    sample_patts = []  # List to store sampled patterns

    # Generating sample patterns
    for _ in range(sample_size):
        x = Decimal(random()) * z  # Generating a random value within the total weight
        i = find(w_data, 0, len(w_data), x)  # Finding the index in w_data using binary search
        inst = tab_data[i]  # Getting the corresponding instance set
        patt = set()

        # Randomly selecting elements from the instance set
        for e in inst:
            if random() > 0.5:
                patt.add(e)

        sample_patts.append(patt)  # Adding the sampled pattern to the list

    return sample_patts  # Returning the list of sampled patterns


if __name__ == '__main__':
    databases = ["ORetail", "POWERC", "kddcup99"]#, "SUSY"]
    sample_size = 10000
    sample_patts = []

    # Iterating through different datasets
    for dataname in databases:
        dataset = "Benchmark/Itemset/" + dataname + ".num"
        print(dataname)
        tmps = time.time()
        sample_patts = TwoStep(dataset, sample_size)  # Performing TwoStep sampling
        tmps = time.time() - tmps
        print(f"===  Execution time {dataname}: {tmps} seconds")
