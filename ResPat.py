import time
from mpmath import mp

def ResPat_ExpJ_Damping(dataset, sample_size, tmps, alpha, batchsize, df, idf):
    reservoir = []
    jump = 0
    timestamp = 0
    out_of_timing = False
    nbInstInBatch = 0
    factor = alpha
    
    with open(dataset, 'r') as base:
        line = base.readline()
        timestamp += 1
        
        while line:
            nbInstInBatch += 1
            instance = line.split()
            size = len(instance)
            i = 0
            w = 2 ** size
            
            while i + jump < w:
                i += jump
                pattern = []
                binv = format(int(i), 'b').zfill(size)
                
                # Create pattern based on binary representation
                for l in range(size):
                    if binv[l] == '1':
                        pattern.append(instance[l])
                
                # Find minimum key in reservoir
                j, mk = minkey(reservoir, timestamp, df, sample_size)
                
                # Generate pattern key
                pat_key = mk + mp.rand() * (1 - mk)
                
                # Update reservoir with the pattern
                reservoir = updateSample(reservoir, sample_size, pattern, j, pat_key, timestamp, idf)
                
                # Determine jump size dynamically
                _, mk = minkey(reservoir, timestamp, df, sample_size)
                x = 0
                if mk > 0:
                    x = mp.mpf(int(mp.log(mp.rand()) / mp.log(mk)))
                jump = x + 1
                
                # Check if time limit exceeded
                if time.time() - tmps > 3600:
                    print("Out of timing")
                    out_of_timing = True
                    break
            
            if out_of_timing:
                break
            
            jump = (i + jump) - w
            line = base.readline()
            
            # Check batch size and update factors
            if nbInstInBatch == batchsize:
                timestamp += 1
                if factor > 0:
                    df *= factor
                    idf *= 1 / factor
                nbInstInBatch = 0
    
    return reservoir


def updateSample(reservoir, sample_size, pattern, j, pat_key, timestamp, idf):
    r = mp.mpf(pat_key ** idf)
    
    if len(reservoir) < sample_size:
        reservoir.append((pattern, r, timestamp))
    else:
        reservoir[j] = (pattern, r, timestamp)
    
    return reservoir


def minkey(reservoir, timestamp, df, sample_size):
    if len(reservoir) < sample_size:
        return 0, mp.mpf(0.)
    
    j, mk = 0, mp.mpf(0.9999999999999999)
    
    for i in range(len(reservoir)):
        if reservoir[i][1] < mk:
            mk = reservoir[i][1]
            j = i
            if mk == 0:
                break
    
    mk = mk ** df
    
    if mk == 1:
        mk = 0.9999999999999999
    
    return j, mk


if __name__ == '__main__':
    databases = ["POWERC", "ORetail", "kddcup99", "SUSY"]
    sample_size = 10000
    alpha_values = [0, 0.1, 0.5]
    batchsize = 1000
    
    df = mp.mpf(1)
    idf = mp.mpf(1)
    
    for dataname in databases:
        for alpha in alpha_values:
            streamdata = "Benchmark/Itemset/" + dataname + ".num"
            print(dataname)
            print("sample_size", sample_size) 
    
            tmps = time.time()
            reservoir = ResPat_ExpJ_Damping(streamdata, sample_size, tmps, mp.mpf(alpha), batchsize, df, idf)
            tmps = time.time() - tmps
    
            print(f"===  Execution time {dataname} with damping factor {alpha}: {tmps}")


