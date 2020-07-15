import numpy as np
from scipy.special import logsumexp 
from msrpabstract import SRPAbstract
# from util import greedy

class MSSRP(SRPAbstract):
    """
    Extended the SRPAbstract class
    """
    def __init__(self, p, c, k, M, nsensors, Ks, L=-1, chart = 'srp',mode = 'T2',sample_mode = 'sample'):  
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
        super().__init__(p, c, k, M, nsensors, Ks, L, chart, mode)

        self.sample_mode = sample_mode
    
    def compute_log_LRT(self,a,x,i):
        """
        Compute the log liklihood ratio of 
        Input: 
        - a: sensing vectors
        - x: sensing data, must be in format of p * 1
        """
        c = self.c 
        M = self.M    
        if i == 0:
            E = -2*c*M*x + c**2*M**2
        else:
            E = -2*(c*M-c*M[:,[i-1]])*(x-c*M[:,[i-1]]) + np.multiply(c*M-c*M[:,[i-1]], c*M-c*M[:,[i-1]])
        if self.selectmode == 'indi':
            result = a@E
        elif self.selectmode == 'cs':
            E = 1
        return result


    def compute_index(self,failureModeTopIdx, Ft, pi,r=1):
        """        
        Compute the index function to decide the best sensing allocation
        Input: 
        - x_sample: sampled version of x, must be in format of p * 1 or p * k
        - failureModeTopIdx: The most important failure index
        - mode: Types of monitoring statistics
        """        
        M = self.M
        c = self.c
        p = self.p
        k = self.k
        nsensors = self.nsensors
        x_sample = (np.random.randn(1,p) + np.sum(M[:,failureModeTopIdx]*c*Ft[failureModeTopIdx], 1)).T
        
        phi = np.zeros(k+1)
        phi[1:] = Ft + pi*(1-Ft)
        phi[0] = np.prod(1-phi[1:])
        
        statistics = np.zeros((p,k, k+1))
        for i in range(k + 1):
            if i == 0:
                statistics[:, :, i] = np.log(phi[i]/phi[1:])  -2*c*M*x_sample + c**2*M**2
            else:
                statistics[:, :, i] =np.log(phi[i]/phi[1:])  -2*(c*M-c*M[:,[i-1]])*(x_sample-c*M[:,[i-1]]) + np.multiply(c*M-c*M[:,[i-1]], c*M-c*M[:,[i-1]])
        statistics_topr = np.sum(statistics[:,failureModeTopIdx,:],1)
        statistics_topr_sort= np.sort(statistics_topr, axis = 0)
        kmax = np.argmax(np.sum(statistics_topr_sort[:nsensors,:] ,0))
        sensingIdx = np.argsort(-statistics_topr[:,kmax])[:nsensors]
            
        return sensingIdx


    