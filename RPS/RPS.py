__author__ = ""

import sys
from random import random, sample, randint
import scipy.special as sc
from decimal import Decimal
from math import log
from math import pow
import time

class RPS:
    def __init__(self,
                 sample_size:int,
                 dampingFactor:float, 
                 utilityMeasure:str, 
                 maxNorm:int,
                 alpha:float,
                 patternLanguage:str,
                 batchsize:int,
                 labled,
                 weightedItems,
                 classification,
                 predict_duration,
                 learning_duration,
                 model_classifier):
        
        """
        Initialize the RPS with given parameters.
        
        Args:
        - sample_size: Number of samples to be maintained in the reservoir.
        - dampingFactor: Damping factor applied to timestamps.
        - utilityMeasure: Measure used for utility calculation (area, freq, decay).
        - maxNorm: Maximum norm value for subsequences.
        - alpha: Alpha parameter for utility measure.
        - patternLanguage: Type of pattern language (Sequence or Itemset).
        - batchsize: Size of each batch for sampling.
        - labled: Flag indicating if the data is labeled.
        - weightedItems: Flag indicating if items are weighted.
        - classification: Flag indicating if the task is classification.
        - predict_duration: Duration for prediction in classification.
        - learning_duration: Duration for learning in classification.
        - model_classifier: Model classifier instance for classification.
        """
        self.itemsetDelimiter= '-1'
        self.delimiteurSequence= '-2'
        self.delimWeightItems = ":"
        self.sample_size = sample_size
        self.dampingFactor = dampingFactor
        self.utilityMeasure = utilityMeasure 
        self.maxNorm = maxNorm
        self.alpha = alpha
        self.cnk_matrix = [[1]]
        self.reservoir = []
        self.patternLanguage = patternLanguage
        self.batchsize = batchsize
        self.labled = labled
        self.weightedItems = weightedItems
        self.classification = classification
        self.predict_duration = predict_duration
        self.time_to_learn = True
        self.learning_duration = learning_duration
        self.model_classifier = model_classifier
        
        if self.classification and not (self.patternLanguage == "Sequence"):
            raise ValueError("Classification is only supported for Sequence pattern language.")

    #Position set
    def positionSet(self, sequence, itemset):
        ps,i=[], 0
        x=len(sequence)
        while i<x:
            intersec= sequence[i].intersection(itemset)
            if len(intersec)!=0:
                k,chaine=i+1, intersec
                while k < x and not chaine.issubset(sequence[k].intersection(itemset)):
                    k+=1
                if k==x:
                    ps.append(i)
            i+=1
        return ps
    
    
    def setOfSubsets(self, ens):
        p = []
        i, imax = 1, 2**len(ens)-1
        while i <= imax:
            s = []
            j, jmax = 0, int(log(i,2))
            while j <= jmax:
                if (i>>j)&1 == 1:
                    s.append(ens[j])
                j += 1
            p.append(s)
            i += 1 
        return p
    
    
    def nbSousSeqCSSampling(self, sequence):
        """
        Algo dynamique
    
        - nombre de sous séquences dans `sequence` de taille <=k (par normes)
        - retourne le nombre pour chaque norme
        """
        if self.maxNorm==0 or sequence==[]:
            return [[1]]
        else:
            nt=sum([len(its) for its in sequence])
            k=min(self.maxNorm,nt)
            T=[[1],[1]]
            R=[[0],[0]]
            for i in range(1,k+1):
                T[0].append(1)
                T[1].append(T[1][i-1]+self.combinCnk(len(sequence[0]), i))
                R[0].append(0)
                R[1].append(0)
            for i in range(2,len(sequence)+1):
                R.append([0])
                T.append([1])
                ps=self.positionSet(sequence[:i-1], sequence[i-1])
                sousEnsPS=self.setOfSubsets(ps)
                for j in range(1,k+1):
                    T[i].append(int(0))
                    R[i].append(int(0))
                    for u in range(len(sousEnsPS)):
                        intersecMulti=sequence[:i-1][sousEnsPS[u][0]]
                        v=1
                        while v<len(sousEnsPS[u]):
                            intersecMulti=intersecMulti.intersection(sequence[:i-1][sousEnsPS[u][v]])
                            v+=1
                        intersecMulti=sequence[i-1].intersection(intersecMulti)
                        m = min(sousEnsPS[u])
                        kmax = len(intersecMulti)
                        for v in range(1,j+1):
                            R[i][j] += pow(-1,len(sousEnsPS[u])+1)*T[m][j-v]*self.combinCnk(kmax, v)
                    for v in range(min([len(sequence[i-1]),j])+1):
                        T[i][j] += T[i-1][j-v]*self.combinCnk(len(sequence[i-1]), v)
                    T[i][j] -= R[i][j]
            return T[-1][0:1]+[T[-1][i]-T[-1][i-1] for i in range(1,len(T[-1]))]
    
        
    
    def  compute_Cnk(self, j):
        for i in range(len(self.cnk_matrix), j+1):
            self.cnk_matrix.append([1]+[self.cnk_matrix[i-1][min(i-k, k-1)]+self.cnk_matrix[i-1][min(i-1-k, k)] for k in range(1, int(i/2)+1)])
        return self.cnk_matrix
    
    
    def combinCnk(self, j, l):
        if l>j:
            return 0
        if j >= len(self.cnk_matrix):
            self.cnk_matrix = self.compute_Cnk(j)
        if j<0 or l<0:
            print(j,l, self.cnk_matrix, self.cnk_matrix[j][min(j-l, l)])
        return self.cnk_matrix[j][min(j-l, l)]
        
    def combinCnk1(self, n, k, tabCnk):
        if k>n or n==0: return 0
        return tabCnk[n][k]
    
    def computeCnkFast(self, n: int, M: int):
        for i in range(len(self.cnk_matrix), n+1):
            self.cnk_matrix.append([1]+[self.cnk_matrix[i-1][k-1]+self.cnk_matrix[i-1][k] for k in range(1, min(M,i))])
            if i<=M:
                self.cnk_matrix[i].append(1)
            else:
                self.cnk_matrix[i].append(self.cnk_matrix[i-1][M-1]+self.cnk_matrix[i-1][M])
                
    def sumUtility(self, M):
        som = 0.0
        if self.utilityMeasure == "area":
            for i in range(len(M)):
                som+=M[i]*(i+1)
        elif self.utilityMeasure == "freq":
            som = sum(M)
        elif self.utilityMeasure == "decay":
            for i in range(len(M)):
                som+=M[i]*pow(self.alpha,i+1)
        return Decimal(som)
        
    def k_norme(self, tabNorm, tab):
        """
        - tabNorm: contains the number of subsequences of norm l for l in [1..maxNorm]
        - tab: contains the weight of each norm and reused when rejection
        
        Compute the weight of each norm according the norm-based utility 
        and draw one of them proportionally to its weight
        """
        if len(tab)==0:
            som=0
            if self.utilityMeasure=="freq":
                for val in tabNorm:
                    som+=val
                    tab.append(som)
            elif self.utilityMeasure=="area":
                for l in range(len(tabNorm)):
                    som+=tabNorm[l]*(l+1)
                    tab.append(som)
            elif self.utilityMeasure=="decay":
                for l in range(len(tabNorm)):
                    som+=tabNorm[l]*pow(self.alpha,l+1)
                    tab.append(som)
        
        #print(tab)
        i,j=0,len(tab)
        randVal = random()*tab[j-1]
        k = self.find(tab,i,j,randVal)
        return k+1
        
    
                        
    def drawSubSequenceExact(self, sequence2,tabNorme):
        subSequence=[]
        rejet=True
        Tab=[]
        maSeq=[]
        for itemset in sequence2:
            for item in itemset:
                maSeq.append(item)
            maSeq.append(self.itemsetDelimiter)
        Tab[:]=[i for i in range(len(maSeq)) if str(maSeq[i])!=self.itemsetDelimiter]
        while rejet==True:
            tab=[]
            selectedNorm = self.k_norme(tabNorme, tab)
            T=[]
            T[:]=list(Tab)
            X=[]
            for i in range(selectedNorm):            
                m= randint(1,len(T))-1
                X.append(T[m])
                T.remove(T[m])
            f=0            
            subSequence, itemset =[],set()
            while f< len(maSeq):
                if f in X:
                    itemset.add(maSeq[f])
                elif maSeq[f]==self.itemsetDelimiter:
                    subSequence.append(itemset)
                    itemset = set()
                f+=1
            
            n=len(subSequence)-1
            rejet=False
            if subSequence != sequence2:
                while n>0 and rejet==False:
                    if len(subSequence[n]) != 0:
                        k=n-1
                        while k>=0 and rejet==False and len(subSequence[k])!=0:
                            if len(subSequence[n]-sequence2[k]) != 0:
                                k-=1
                            else:
                                subSequence =[]
                                rejet=True
                                #totalRejet+=1
                                selectedNorm = self.k_norme(tabNorme, tab)
                    n-=1
        return [itemset for itemset in subSequence if len(itemset)>0]#,totalRejet
        
    
    
    def find(self, tab,i,j,x):
        m=int((i+j)/2)
        if m==0 or (tab[m-1]<x and x<=tab[m]):
            return m
        if tab[m]<x:
            return self.find(tab,m+1,j,x)
        return self.find(tab,i,m,x)
    
    
    def temporalBias(self, timestamp):
        if self.dampingFactor==0:
            return 1
        else:
            x = Decimal(self.dampingFactor*timestamp)
            return x.exp()
    
    def getLanguage(self, line):
        if self.patternLanguage == "Sequence":
            line = line.replace(self.delimiteurSequence,"").replace("\n","")
            return [set(i.split()) for i in line.split(self.itemsetDelimiter+' ')[:-1]]
        elif self.patternLanguage == "Itemset":
            return line.split()

    def weighted_norme_utility(self, tabNorm):
        """
        - tabNorm: contains the number of subsequences of norm l for l in [1..maxNorm]
        
        Compute the weight of each norm according the norm-based utility 
        and draw one of them proportionally to its weight
        """
        tab = []
        som=0
        if self.utilityMeasure=="freq":
            for val in tabNorm:
                som+=val
                tab.append(som)
        elif self.utilityMeasure=="area":
            for l in range(len(tabNorm)):
                som+=tabNorm[l]*(l+1)
                tab.append(som)
        elif self.utilityMeasure=="decay":
            for l in range(len(tabNorm)):
                som+=tabNorm[l]*pow(self.alpha,l+1)
                tab.append(som)
        return tab

    def getWeightedInstance(self, instance0):
        new_l = ""
        maxPatternNorm = 0
        tabNorme = []
        w = Decimal(0.)
        if self.patternLanguage == "Sequence":
            instance0 = instance0.replace(self.delimiteurSequence,"").replace("\n","")
            instance0 = [set(i.split()) for i in instance0.split(self.itemsetDelimiter+' ')[:-1]]
        else:
            instance0 = instance0.split()
        if self.labled:
            new_l = list(instance0[0])[0]
            instance0 = instance0[1:]
        if self.patternLanguage=="Itemset":
            if self.utilityMeasure in ["HUI", "HAUI"]:
                w_instance = Decimal(0.)
                len_instance = len(instance0)
                maxPatternNorm = len_instance
                instance = []
                for info in instance0:
                    info = info.split(self.delimWeightItems)
                    w_instance += Decimal(info[1])
                    instance.append([info[0], w_instance])
                if self.maxNorm=="Infty" and self.utilityMeasure=="HUI":
                    w = w_instance*Decimal(2**(len_instance - 1))
                elif self.maxNorm!="Infty" and self.utilityMeasure=="HUI":
                    maxPatternNorm = self.maxNorm
                    for l in range(1, min(len(instance), self.maxNorm) + 1):
                        w += (w_instance * self.combinCnk(len(instance) - 1, l - 1))
                elif self.maxNorm!="Infty" and self.utilityMeasure=="HAUI":
                    maxPatternNorm = self.maxNorm
                    for l in range(1, min(len(instance), self.maxNorm) + 1):
                        w += (w_instance * self.combinCnk(len(instance) - 1, l - 1)) / l
                else:
                    sys.exit(0)
                #return instance, [], w, maxPatternNorm
                instance0 = list(instance)
            else:
                if self.maxNorm=="Infty":
                    w = 2**len(instance0) - 1
                else:  
                    tabNorm = []
                    j = len(instance0)
                    for l in range(1,min(j,self.maxNorm)+1):
                        tabNorm.append(self.combinCnk(j, l))
                    tabNorme = self.weighted_norme_utility(tabNorm)
                    if len(tabNorme)>0:
                        w = self.sumUtility(tabNorme)
                    #return instance0, w_tab, Decimal(sum(w_tab)), ""
        elif self.patternLanguage=="Sequence":
            tabNorme = self.nbSousSeqCSSampling(instance0)
            if len(tabNorme)>0:
                w = self.sumUtility(tabNorme[1:])
            #return instance0, tabNorme, w, ""
        return instance0, new_l, tabNorme, w, maxPatternNorm
    
    
    
    def getInstance(self, instance0):
        new_l = ""
        if self.patternLanguage == "Sequence":
            instance0 = instance0.replace(self.delimiteurSequence,"").replace("\n","")
            instance0 = [set(i.split()) for i in instance0.split(self.itemsetDelimiter+' ')[:-1]]
        else:
            instance0 = instance0.split()
        if self.labled:
            new_l = list(instance0[0])[0]
            instance0 = instance0[1:]
        if self.patternLanguage=="Itemset":
            if self.utilityMeasure in ["HUI", "HAUI"]:
                instance = []
                for info in instance0:
                    info = info.split(self.delimWeightItems)
                    instance.append(info)
                instance0 = list(instance)
        return instance0, new_l

    
    

    def drawPatternIP(self, batch, w_batch, batch_tabNorme):
        r = random() * float(w_batch[-1])
        i = self.find(w_batch, 0, len(w_batch), r)
        instance = batch[i]
        w_ins_tab = batch_tabNorme[i]
        w = w_batch[i]
        if i>0:
            w -= w_batch[i-1]
        
        pattern = []
        if len(w_ins_tab) == 0:
            for e in instance:
                if random()>=0.5:
                    pattern.append(e)
        else:
            l = self.draw_lengthIP(w_ins_tab, w)
            #print(l, w_ins_tab, w, instance)
            pattern = sample(instance, l)
        return str(pattern)
    
    def draw_lengthIP(self, w_ins_tab, w):
        z = float(w)*random()
        for l in range(1, len(w_ins_tab)+1):
            z -= w_ins_tab[l-1]
            if z<=0:
                return l
        return len(w_ins_tab)

    def drawPattern(self, batch, w_batch, batch_tabNorme, maxPatternNorm, z_batch):
        if self.patternLanguage == "Sequence":
            r = random() * float(z_batch)
            i = self.find(w_batch, 0, len(w_batch), r)
            instance = batch[i]
            return self.drawSubSequenceExact(instance, batch_tabNorme[i][1:])
        else:
            if self.weightedItems:
                return self.drawPatternWIP(batch, maxPatternNorm, w_batch)
            else:
                return self.drawPatternIP(batch, w_batch, batch_tabNorme)
                                
        

    def draw_lengthWIP(self, trans, maxPatternNorm, w):
        z = w*Decimal(random())
        for l in range(1, maxPatternNorm+1):
            z -= trans[-1][1]*self.combinCnk(len(trans)-1, l-1)
            if z<=0:
                return l
        return len(trans)

    def drawPatternWIP(self, batch, maxPatternNorm, w_batch):
        r = random() * float(w_batch[-1])
        i = self.find(w_batch, 0, len(w_batch), r)
        instance = batch[i]
        w = w_batch[i]
        if i>0:
            w -= w_batch[i-1]
        l = self.draw_lengthWIP(instance, maxPatternNorm, w)
        pattern = []
        i = len(instance)
        x = Decimal(random())*instance[i-1][1]*self.combinCnk(i-1, l-1)
        agg_util = 0
        while l>0:
            m = self.find_index(x, l-1, i, l, instance, agg_util)
            pattern = [instance[m][0]] + pattern
            agg_util += instance[m][1]
            if m>0:
                agg_util -= instance[m-1][1]
            l -= 1
            i = m
            if l>0:
                x = Decimal(random())*(instance[i-1][1]+agg_util)*self.combinCnk(i-1, l-1)
        return pattern


    def find_index(self, x, j, i, l, wtrans, agg_util):
        m = int((j+i)/2)
        b_sup = (wtrans[m][1]+agg_util)*self.combinCnk(m, l-1)
        b_inf = (wtrans[m-1][1]+agg_util)*self.combinCnk(m-1, l-1)
        if b_inf < x and x <= b_sup:
            return m
        if b_sup < x:
            return self.find_index(x, m+1, i, l, wtrans, agg_util)
        return self.find_index(x, j, m, l, wtrans, agg_util)

    def Sampler(self, streamdata):
        reservoir = []
        sum_w_res = Decimal(0.)
        timestamp =0
        has_pred = False
        last_time_learning = 0
        with open(streamdata, 'r') as base:
            batch = []
            w_batch = []
            z_batch = 0
            batch_tabNorme = []
            batch_data = []
            line = base.readline()
            while line:
                if len(w_batch) ==0:
                    timestamp += 1
                if len(w_batch) != self.batchsize:
                    instance, new_l, tabNorme, w, maxPatternNorm = self.getWeightedInstance(line)
                    
                    batch_data.append((instance, new_l))
                    x = self.temporalBias(timestamp)
                    sum_w_res += w*x
                    z_batch += w
                    batch.append(instance)
                    w_batch.append(z_batch)
                    batch_tabNorme.append(tabNorme)
                line = base.readline()
                if (not line) or len(w_batch) == self.batchsize:
                    if not self.classification or self.time_to_learn:
                        start_time = time.time()
                        if len(reservoir) == 0:
                            for _ in range(self.sample_size):
                                reservoir.append(self.drawPattern(batch, w_batch, batch_tabNorme, maxPatternNorm, z_batch)) #((subSequenceStr, ))
                        else:
                            #print(z_batch, x, sum_w_res, line, len(reservoir))
                            p_success = float( (z_batch*x) / sum_w_res)
                            q = random()
                            if p_success >= q: 
                                k = 1+int(round(sc.bdtrik(q, self.sample_size-1, p_success, out=None),0))
                                E = sample(range(self.sample_size), k)
                                for j in E:
                                    reservoir[j] = self.drawPattern(batch, w_batch, batch_tabNorme, maxPatternNorm, z_batch) #(subSequenceStr, )
                        elapsed_time = time.time() - start_time
                        if not self.classification:
                            print(f"Execution time(Batch_{timestamp}): {elapsed_time} seconds")
                    if self.classification:
                        if len(batch_data)>0:
                            if self.time_to_learn:
                                print(f"Learning(Batch_{timestamp})...")
                                self.model_classifier.reservoir_learning(batch_data, reservoir)
                                if last_time_learning + self.learning_duration == timestamp:
                                    self.time_to_learn = False
                                    last_time_learning = timestamp
                            else:
                                drift, msg, _ = self.model_classifier.reservoir_predict(batch_data, reservoir)
                                to_print = f"Accuracy(Batch_{timestamp})="+str(self.model_classifier.accuracy[-1])
                                has_pred = True
                                if drift or last_time_learning + self.predict_duration == timestamp:
                                    self.time_to_learn = True
                                    last_time_learning = timestamp
                                    to_print += str(msg)
                                    #has_learnt = False
                                #print(reservoir)
                                print(f"{to_print}")
                    del batch
                    del w_batch
                    del batch_tabNorme                    
                    del batch_data
                    batch_data = []
                    batch = []
                    w_batch = []
                    z_batch = 0
                    batch_tabNorme = []
                    
        return reservoir, has_pred

    


import numpy as np
from sklearn.metrics import accuracy_score
import copy
from .kswin import KSWIN

class Classifier:
    def __init__(self, patternLanguage, model_algo):
        self.empty_model = copy.deepcopy(model_algo)
        self.current_model = copy.deepcopy(model_algo)
        self.isFitted = False
        self.accuracy = []
        self.precision = []
        self.recall = []
        self.f1_score = []
        self.matthews = []
        self.average_precision = []
        self.roc_auc = []
        self.past_labels = set()
        self.current_labels = set()
        self.patternLanguage = patternLanguage
        self.number_of_predicted_instances = 0
        self.number_of_good_predicted_instances = 0
        self.len_old_attributes = 0
        self.len_current_attributes = 0
        self.batch_number = 0
        self.adwin = KSWIN(alpha=0.1)
        self.confusion_matrix = []
        
        
    def reservoir_learning(self, batch_train, reservoir):
        for (_, y) in batch_train:
            self.current_labels.add(y)
       
        if not ((len(self.current_labels)>1) and (self.past_labels == self.current_labels)):
            self.current_model = copy.deepcopy(self.empty_model)
            self.past_labels.update(self.current_labels)
    
        if len(self.past_labels)>1 :
            if len(batch_train)>0:
                self.pfit(batch_train, reservoir)
        
    def reservoir_predict(self, batch_test, reservoir):
        drift = False
        msg = ""
        if len(batch_test) >0:
            acc, new_labels, labels = self.predict(batch_test, reservoir)
            self.adwin.update(acc)
            if self.adwin.drift_detected:
                del self.current_model
                msg = "drift_detected adwin"
                self.current_model = copy.deepcopy(self.empty_model)
                drift = True
            if not new_labels.issubset(self.current_labels):
                msg = "drift_detected new_labels"
                self.current_model = copy.deepcopy(self.empty_model)
                self.past_labels.update(self.current_labels)
                drift = True
            if drift:
                for (_, y) in batch_test:
                    self.current_labels.add(y)
                self.past_labels.update(self.current_labels)
                self.pfit(batch_test, reservoir)
            if not drift:
                msg = acc
        return drift, msg, labels
        
    def pfit(self, batch_data, reservoir):
        instances, our_labels = [], []
        for instance, label_inst in batch_data:
            bin_inst = []
            for pattern in reservoir:
                bin_inst.append(self.binaryReleation(pattern, instance))
            instances.append(bin_inst)
            our_labels.append(label_inst)
        self.current_model.partial_fit(instances, our_labels, classes=np.unique(list(self.current_labels)))

    def predict(self, batch_data, reservoir):
        instances, true_labels = [], []
        for instance, label_inst in batch_data:
            bin_inst = []
            for pattern in reservoir:
                bin_inst.append(self.binaryReleation(pattern, instance))
            instances.append(bin_inst)
            true_labels.append(label_inst)
        pred_labels = self.current_model.predict(instances)
        self.number_of_predicted_instances += len(instances)
        acc = accuracy_score(true_labels, pred_labels)
        self.number_of_good_predicted_instances += acc*len(instances)
        self.accuracy.append(acc)
        labels = list(self.current_labels)
        return acc, set(true_labels), labels
    

    #Check if a sequence is a subsequence of another sequence
    def isSubSequence(self, sousSeq, sequence):
        if len(sousSeq)>len(sequence):
            return 0 #False
        i,j,ok,taille2=0,0,True, len(sequence)
        #print(sousSeq, sequence)
        while i<len(sousSeq) and ok:
            while j<len(sequence) and not sousSeq[i].issubset(sequence[j]): 
                j+=1
            if j==taille2:
                ok=False    
            else:
                j+=1
            i+=1
        
        return 1 if ok==True else 0

    def is_subset_of(self, pattern, instance):
        """Check if 'pattern' is a subset of 'instance'."""
        return 1 if pattern.issubset(instance) else 0
    
    def binaryReleation(self, pattern, instance):
        if self.patternLanguage == "Sequence":
            return self.isSubSequence(pattern, instance)
        elif self.patternLanguage == "Itemset":
            return self.is_subset_of(pattern, instance)
 
    

