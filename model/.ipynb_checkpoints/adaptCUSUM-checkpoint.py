import numpy as np
from scipy.special import logsumexp 
from model.srpabstract import SRPAbstract
import pdb
class AdaptCUSUM(SRPAbstract):
    """
    Extended the SRPAbstract class
    """
    def __init__(self, p, c, k, M, nsensors, Ks,mumin = 2, delta=0.5, L=-1,  chart = 'srp',mode = 'T2',sample_mode = 'sample', selectmode = 'indi',decisionchart=1):  
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
        super().__init__(p, c, k, M, nsensors, Ks, L, chart, mode, selectmode, decisionchart)
        self.delta = delta
        self.mumin = mumin
        
    
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
        delta = self.delta
        nsensors = self.nsensors
        mumin = self.mumin
        sequential_statistics_1 = np.zeros((Tmax,k))
        sequential_statistics_2 = np.zeros((Tmax,k))        
        sequential_statistics_topRsum = np.zeros((Tmax))
        individualS = np.random.randn(k)
        failureModeTopIdx = np.argsort(-individualS)[:Ks]
        A = np.zeros((Tmax,Ks)).astype(int)
        
        A[0,:] = failureModeTopIdx;
        for i in range(1,Tmax-1):
            sequential_statistics_1[i,:] = sequential_statistics_1[i-1,:] + delta;
            sequential_statistics_2[i,:] = sequential_statistics_2[i-1,:] + delta;
            tmp1 = np.zeros((p));
            tmp2 = np.zeros((p));
            tmp1[A[i,:]] = sequential_statistics_1[i-1,A[i,:]] + mumin*x[i,A[i,:]] - mumin**2/2;

            tmp2[A[i,:]] = sequential_statistics_2[i-1,A[i,:]] - mumin*x[i,A[i,:]] - mumin**2/2;
            sequential_statistics_1[i,A[i,:]] = np.maximum(tmp1[A[i,:]],0);
            sequential_statistics_2[i,A[i,:]] = np.maximum(tmp2[A[i,:]],0);
            tmp = np.maximum(sequential_statistics_1[i,:],sequential_statistics_2[i,:]);
            temp_ind = np.random.permutation(p)
            tmp1 = tmp[temp_ind];
            
            tmpv = np.sort(-tmp1);                                       
            tmpv = -tmpv
            ind = np.argsort(-tmp1);        
            A[i+1,:] = temp_ind[ind[:Ks]]   
            sequential_statistics_topRsum[i] = np.sum(tmpv[:Ks]);
            if L != -1:
                if sequential_statistics_topRsum[i]>L and i>T0:
                    break

        return sequential_statistics_topRsum, A, i, sequential_statistics_1,sequential_statistics_2

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
            srp,a,Tout,b,c= self.compute_monitoring_statistics(idata,T0, L)
            Tmonit_stat[i] = srp
            Tout_all[i] = Tout
        return Tmonit_stat, Tout_all
    


    
    def compute_index(self,failureModeTopIdx,r=1):
        """        
        Compute the index function to decide the best sensing allocation
        Input: 
        - x_sample: sampled version of x, must be in format of p * 1 or p * k
        - failureModeTopIdx: The most important failure index
        - mode: Types of monitoring statistics
        """        
        return failureModeTopIdx


    