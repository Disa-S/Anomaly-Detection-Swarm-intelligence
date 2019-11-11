# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 16:13:08 2019

@author: Disa-S
"""
from pandas import read_csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

dataset = read_csv(r"nyiso_loads.csv", sep=',', header=0, low_memory=False, skiprows = 0)
scaler = MinMaxScaler(feature_range=(0.01, 1)) #normalising between 0.001 and 1 to avoid divisions by zero


class swarmmodel():
    def __init__(self, data, rounds, threshold, swarmsize, alg = "VBA", reinit = "Yes"):
        """
        data : the data employed
        rounds : integer, epoches/rounds used in the swarm
        threshold : float, amount of deviation allowed
        swarmsize : integer, allowed size of the swarm
        alg : optimisation algorithm
        reinit : is the swarm reinitialised each round
        """
        #TODO : lacks M1+M2+ Y sharing
        self.data = data                
        self.datamodified = scaler.fit_transform(np.array(self.data.drop(columns =['Year','Month', 'Day', 'Hr25']))) #columns 0, 1, 2 are dates, so useless for calculations after redundant information is dropped, data is normalised 
        self.allswarms = set() #keeping track of all  swarms thus far
        self.swarmsize = swarmsize
        self.threshold = threshold
        self.rounds = rounds
        self.results = set()
        self.reinit = reinit #whether swarm is reinitialised each round
        self.alg = alg #algorithm choice
        
    def init_swarm(self, swarm): #initialise the original random swarm       
        while True: #keep trying new data points from the dataset until swarm's size is correct
            if len(swarm) == self.swarmsize: return swarm
            randomdatapoint = np.random.randint(0,len(self.datamodified)) #we only care about the index of the meter, not the whole matrix 
            if randomdatapoint not in swarm: swarm.add(randomdatapoint)
            else: pass          
 
    def pdf(self, swarmdata): #calculating pdf for given normalised dataset 
        return np.exp(-swarmdata**2./.2)/np.sqrt(2.*np.pi)
    
    def entropy(self,swarmdata):#calculating entropy for given normalised dataset
        return -np.sum(swarmdata*np.log2(swarmdata))
    
    def VBA(self, swarm):
        self.allswarms.add(frozenset(swarm)) #frozenset is hashable, unlike normal set
        consumptions, te = {}, 0.     
        for meter in swarm:
            pdf = self.pdf(self.datamodified[meter])
            #prob = self.datamodified[meter] /self.datamodified[meter].sum()
            entropy= self.entropy(self.datamodified[meter])         
            mean =np.mean(self.datamodified[meter])
            consumptions[meter] = [pdf, mean ,entropy] #all meters calculate mean and entropy, store it in a dict since dict allows access by hash 
            te += entropy       
        smeans = np.zeros(len(swarm))
        for y, i in enumerate(swarm): #take mean of each hour
            temp = []
            for j in range(len(self.datamodified[0])): temp.append(self.datamodified[i][j])
            smeans[y] = np.mean(temp)
        sentropy = te / len(swarm) #entropy of all meters in swarm
        #spdf = self.pdf(smeans)
        smean = np.mean(smeans)
        lmax, lmaxmeter = 0., -np.inf  #search for local maxima of the swarm
        for meter in swarm:         
            ij = np.sqrt((smean -consumptions[meter][1])**2 + (sentropy - consumptions[meter][2])**2)#sqr((swarm mean - individual mean) ^2 + (swarm entropy - individual entropy) ^2)
            #TODO : ij sometimes returns "nan" ?
            if ij > lmax: #CURRENTLY MAYBE WRONG, since the thresholding is ??
                lmax,lmaxmeter = ij, meter
        if lmax > self.threshold:  return lmaxmeter #if largest deviation is above threshold, flag it
        else: return None
    
    def startswarm(self):
         swarm = set() #set is O(1), using numpy array would make lookup O(n), doesnt matter much since not many lookups       
         for ro in range(self.rounds):
             if ro % 10 == 0: print("Currently on round: %d, Anomalies detected: %d" % (ro, len(self.results)))
             swarm= self.init_swarm(swarm)
             if self.alg == "VBA": lmax = self.VBA(swarm)
             
             if lmax is not None and self.reinit != "Yes":
                 swarm.remove(lmax)
                 self.results.add(lmax)
             elif self.reinit == "Yes":
                swarm = set()
                if lmax is not None: self.results.add(lmax)
                
    def plotresults(self):
        red_dot = mlines.Line2D([], [], color='red', marker='x', linestyle='None',
                          markersize=8, label='Anomaly')
        blue_dot = mlines.Line2D([], [], color='Blue', marker='o', linestyle='None',
                          markersize=8, label='Normal data point')
        for y,i in enumerate(self.datamodified):
            if y not in self.results: plt.scatter(y, np.mean(i),  c ="blue", s = 2)
            else: plt.scatter(y, np.mean(i), marker = "x",  c ="red", s = (14,14)),  plt.annotate(str(y),(y, np.mean(i)), fontsize = 7)
        
        plt.legend(handles = (red_dot, blue_dot),scatterpoints = 1, loc="upper left", title="Anomalies detected using %s \n with t =%5.2f and %d rounds" %(self.alg,self.threshold, self.rounds))
        plt.show()
    
warm = swarmmodel(dataset, 100, 5.8, 30)           
            
warm.startswarm()
warm.plotresults()

print(warm.results)
