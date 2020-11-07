import numpy as np
from scipy.special import logsumexp 
from srpabstract import SRPAbstract

class ExtendedTSSRP(SRPAbstract):
    """
    Extended the SRPAbstract class
    """
    def __init__(self, p, c, k, M, nsensors, Ks, L=-1, chart = 'srp',mode = 'T2',sample_mode = 'sample', selectmode = 'indi',decisionchart=1):  
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
        self.sample_mode = sample_mode
    
    def compute_log_LRT(self,a,x):
        """
        Compute the log liklihood ratio of 
        Input: 
        - a: sensing vectors
        - x: sensing data, must be in format of p * 1
        """
        c = self.c 
        M = self.M    
        E = 2*c*M*x - c**2*M**2         
        if self.selectmode == 'indi':
            result = a@E
        elif self.selectmode == 'cs':
            E = 1
        return result
    
    def log1pexp(r):
        zr = r * 0
        for i,ir in enumerate(zr):
            if ir <100:
                zr[i] = np.log1p(np.exp(ir))
            else:
                zr[i] = ir
        return zr
            

    def compute_index(self,failureModeTopIdx,r=1):
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
        mode = self.mode
        nsensors = self.nsensors
        x_sample = np.zeros((p,k))

        if self.sample_mode == 'sample':
            for ik in range(k):
                x_sample[:,ik] = np.random.randn(1,p) + M[:,ik]*c
        elif self.sample_mode == 'mean':
            for ik in range(k):
                x_sample[:,ik] = M[:,ik]*c
                
        E_sample = 2*c*M*x_sample - c**2*M**2 
        
        if mode == 'T2' or mode == 'T1':
            individualS = np.sum(E_sample[:,failureModeTopIdx],1)
            sensingIdx = np.argsort(-individualS)[:nsensors]  
        elif mode == 'T1_Max':
            E_sample = 2*c*M*x_sample - c**2*M**2 
            S = ExtendedTSSRP.log1pexp(r)[np.newaxis,:]*E_sample
            S_sort= -np.sort(-S, axis = 0)
            kmax = np.argmax(np.sum(S_sort[:nsensors,:] ,0))
            sensingIdx = np.argsort(-S[:,kmax])[:nsensors]
        elif mode == 'T1_App':
            E_sample = 2*c*M*x_sample - c**2*M**2 
            S = ExtendedTSSRP.log1pexp(r)[np.newaxis,failureModeTopIdx]*E_sample[:,failureModeTopIdx]
            p = S.shape[0]
            f= lambda a: logsumexp(S.T@a)      
            sensingIdx = greedy(f,p,nsensors)
            sensingIdx = sensingIdx.astype(int)
        return sensingIdx


    