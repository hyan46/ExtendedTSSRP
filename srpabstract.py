import numpy as np
from scipy.special import logsumexp 

class SRPAbstract:
    """
    The abstract class for Thompas sampling SRP statistics
    """
    
    def __init__(self, p, c, k, M, nsensors, Ks, L=-1, chart = 'srp',mode = 'T2',selectmode='indi'):        
        """
        srp is the main class of the library 
        Input: 
        - p: Number of dimensions
        - c: scale vector, Target meanshift is c * M
        - k: Number of failuer Mode
        - M: Failure Mode Mean Matrix of k failure modes: p * k 
        - nsensors: number of selected sensors     
        - Ks: Number of selected failure mode
        - L: control limit, set to -1 if not initialized yet.
        """
        self.p = p
        self.c = c
        self.k = k
        self.M = M
        self.nsensors = nsensors
        self.Ks = Ks
        self.chart = chart
        self.mode = mode 
        self.selectmode = selectmode
        
    def compute_log_LRT(self,a,x):
        """
        Compute the log liklihood ratio of 
        Input: 
        - a: sensing vectors
        - x: sensing data, must be in format of p * 1
        """
        pass

    def compute_index(self,failureModeTopIdx,r=1,mode='T2'):
        """        
        Compute the index function to decide the best sensing allocation
        Input: 
        - x_sample: sampled version of x, must be in format of p * 1 or p * k
        - failureModeTopIdx: The most important failure index
        - mode: Types of monitoring statistics
        """        
        pass
    
    def compute_monitoring_statistics(self,x,T0,L):
        """        
        Compute monitoring statistics
        Input: 
        - x: input data
        - T0: time of change
        - L: control limit
        """        
        Tmax = x.shape[0]
        k = self.k
        Ks = self.Ks                
        M = self.M
        c = self.c
        p = self.p
        nsensors = self.nsensors

        sequential_statistics = np.zeros((Tmax,k))
        sequential_statistics_topRsum = np.zeros((Tmax))
        individualS = np.random.randn(k)
        failureModeTopIdx = np.argsort(-individualS)[:Ks]
        failure_mode_history = np.zeros((Tmax,Ks))
        
        if self.selectmode == 'indi':
            a = np.zeros(p)
            sensor_selection_history = np.zeros((Tmax,nsensors)); 
        elif self.selectmode == 'cs':
            a = np.zeros(p)
            sensor_selection_history = np.zeros((Tmax,nsensors,p)); 
        
        for i in range(Tmax):
            if self.chart == 'srp':
                sequential_statistics[i,:] = np.log1p(np.exp(sequential_statistics[i-1,:])) + self.compute_log_LRT(a,x[[i],:].T)
            elif self.chart == 'cusum':
                sequential_statistics[i,:] = np.maximum(sequential_statistics[i-1,:] + self.compute_log_LRT(a,x[[i],:].T),0)
                
            failureModeTopIdx = np.argsort(-sequential_statistics[i,:])[:Ks]  
            sensingSel = self.compute_index(failureModeTopIdx,r=sequential_statistics[i-1,:])
            if self.selectmode == 'indi':
                a = np.zeros(p)
                a[sensingSel] = 1
            elif self.selectmode == 'cs':            
                a = sensingSel
                
            sensor_selection_history[i] = sensingSel
            failure_mode_history[i] = failureModeTopIdx
            if self.chart == 'srp' and self.mode == 'T1':
                sequential_statistics_topRsum[i] = logsumexp(sequential_statistics[i,failureModeTopIdx])
            elif self.chart == 'srp' and self.mode == 'T2':
                sequential_statistics_topRsum[i] = np.sum(sequential_statistics[i,failureModeTopIdx])
            elif self.chart == 'cusum':                
                sequential_statistics_topRsum[i] = np.sum(sequential_statistics[i,failureModeTopIdx])
            
            if L != -1:
                if sequential_statistics_topRsum[i]>L and i>T0:
                    break
        return sequential_statistics_topRsum, sensor_selection_history, failure_mode_history, i, sequential_statistics

    def compute_monitor_batch(self,x, T0, L):
        """        
        Compute monitoring statistics for batch of samples
        Input: 
        - x: Batched input data
        - T0: time of change
        - L: control limit
        """                
        nbatch = x.shape[0]
        Tmax = x.shape[1]
        Tmonit_stat = np.zeros((nbatch,Tmax))
        Tout_all = np.zeros(nbatch)
        for i,idata in enumerate(x):
            srp,a,b,Tout,c= self.compute_monitoring_statistics(idata,T0, L)
            Tmonit_stat[i] = srp
            Tout_all[i] = Tout
        return Tmonit_stat, Tout_all
    
