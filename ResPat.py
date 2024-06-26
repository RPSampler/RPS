
import time
from random import random
from decimal import Decimal

def ResPat_ExpJ_Damping(dataset, sample_size, tmps, alpha, batchsize):
    reservoir = []
    jump = 0
    timestamp = 0
    nbInstInBatch = 0
    out_of_timing = False
    with open(dataset, 'r') as base:
        line = base.readline()
        timestamp += 1
        while line:
            nbInstInBatch += 1
            instance = line.split()
            size = len(instance)
            i = 0
            w = 2**size
            while i+jump < w:
                i += jump
                pattern = []
                binv =format(i, 'b')
                for l in range(len(binv)):
                    if int(binv[l])==1:
                        pattern.append(instance[size-len(binv):][l])
                j, mk = minkey(reservoir, timestamp, alpha, sample_size)
                pat_key = mk + Decimal(random())*(1-mk)
                reservoir = updateSample(reservoir, sample_size, pattern, j, pat_key, timestamp)
                x =int( Decimal(random()).ln()/(mk.ln()) )
                jump = x + 1
                if time.time()-tmps > 3600:
                    print("Out of timing")
                    out_of_timing = True
                    break
            if out_of_timing:
                break
            jump = (i+jump) - w
            line = base.readline()
            if nbInstInBatch == batchsize:
                timestamp += 1
                nbInstInBatch = 0
    return reservoir


def updateSample(reservoir, sample_size, pattern, j, mk, timestamp):
    c = Decimal(alpha)*Decimal(timestamp)
    r = mk**Decimal(1/c.exp())
    if len(reservoir)<sample_size:
        reservoir.append((pattern, r, timestamp))
    else:
        reservoir[j] = (pattern, r, timestamp)
    return reservoir

def minkey(reservoir, timestamp, alpha, sample_size):
    if len(reservoir) < sample_size:
        return 0, Decimal(0.)
    j, mk = 0, Decimal(0.9999999999)
    c = Decimal(alpha)*Decimal(timestamp)
    for i in range(len(reservoir)):
        if Decimal(reservoir[i][1])**c.exp() < mk:
            mk = Decimal(reservoir[i][1])**c.exp()
            j = i
    return j, mk

if __name__ == '__main__':
    databases = ["ORetail", "kddcup99", "POWERC", "SUSY"]
    sample_size = 10000
    alpha = 0.1 # damping factor
    batchsize = 1000
    for dataname in databases:
        streamdata = "Benchmark/Itemset/"+dataname+".num"
        print(dataname)
        print("sample_size",sample_size) 

        tmps = time.time()
        reservoir = ResPat_ExpJ_Damping(streamdata, sample_size, tmps, Decimal(alpha), batchsize)
        tmps = time.time()-tmps

        print(f"===  Execution time {dataname} with damping factor {alpha}: {tmps}")

        
