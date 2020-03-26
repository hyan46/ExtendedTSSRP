import numpy as np
from scipy.special import logsumexp 

class SRPAbstract:
    """
    The abstract class for Thompas sampling SRP statistics
    """
    
    def __init__(self, p, c, k, M, nsensors, Ks, L=-1):        
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

        srp = np.zeros((Tmax,k))
        a = np.zeros(p)
        srp_sum = np.zeros((Tmax))
        cumsum_sum = np.zeros((Tmax))
        individualS = np.random.randn(k)
        failureModeTopIdx = np.argsort(-individualS)[:Ks]
        sensor_selection_history = np.zeros((Tmax,nsensors)); 
        failure_mode_history = np.zeros((Tmax,Ks))
        for i in range(Tmax):
            srp[i,:] = np.log1p(np.exp(srp[i-1,:])) + self.compute_log_LRT(a,x[[i],:].T)
            failureModeTopIdx = np.argsort(-srp[i,:])[:Ks]  
            sensingIdx = self.compute_index(failureModeTopIdx,r=srp[i-1,:])
            a = np.zeros(p)
            a[sensingIdx] = 1
            sensor_selection_history[i,:] = sensingIdx
            failure_mode_history[i,:] = failureModeTopIdx
            srp_sum[i] = logsumexp(srp[i,failureModeTopIdx])
            if L != -1:
                if srp_sum[i]>L and i>T0:
                    break
        return srp_sum, sensor_selection_history, failure_mode_history, i

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
            srp,a,b,Tout= self.compute_monitoring_statistics(idata,T0, L)
            Tmonit_stat[i] = srp
            Tout_all[i] = Tout
        return Tmonit_stat, Tout_all
    