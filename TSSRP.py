import numpy as np
from scipy.special import logsumexp 
from srpabstract import SRPAbstract
import pdb
class TSSRP(SRPAbstract):
    """
    Extended the SRPAbstract class
    """
    def __init__(self, p, c, k, M, nsensors, Ks, L=-1, chart = 'srp',mode = 'T2',selectmode='indi',decisionchart = 1):  
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

            
    def compute_log_LRT(self,a,x):
        """
        Compute the log liklihood ratio of 
        Input: 
        - a: sensing vectors
        - x: sensing data, must be in format of p * 1
        """
        c = self.c 
        M = np.eye(self.p)
        E = 2*c*M*x - c**2*M**2
        return a@E


    def compute_index(self,failureModeTopIdx,r=1):
        """        
        Compute the index function to decide the best sensing allocation
        Input: 
        - x_sample: sampled version of x, must be in format of p * 1 or p * k
        - failureModeTopIdx: The most important failure index
        - mode: Types of monitoring statistics
        """        

        c = self.c
        nsensors = self.nsensors
        if self.mode == 'T2':
            sensingIdx = failureModeTopIdx
        elif self.mode == 'T1':
            individualS = np.sum(np.exp(r[failureModeTopIdx]))
            sensingIdx = np.argsort(-individualS)[:nsensors]  
        elif self.mode == 'full':
            sensingIdx = np.arange(nsensors)
        return sensingIdx


    