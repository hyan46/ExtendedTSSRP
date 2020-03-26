import pdb
import pandas as pd
import numpy as np
from scipy.optimize import bisect
from tqdm import tnrange, tqdm_notebook, tqdm

class spc:
    """
    spc is the main class of the library for statistical process control
    """
    
    def __init__(self, monitor_statistics, data_gen_func0,data_gen_func1, L = -1):        
        """
        initialization
        Input: 
        - monitor_statistics: given input of n_batch * Tmax * dimensions, return the monitoring statistics of size n_batch * Tmax denote 
        - data_gen_func0: Generate normal samples, return n_batch * Tmax * dimensions:
        Examples: 
        def data_gen_func0(n_batch, Tmax, seed):
            np.random.seed(seed)
            data = np.random.randn(n_batch, Tmax, 10)
            return data
        - data_gen_func1: Generate abnormal samples, return n_batch * Tmax * dimensions
        Examples: 
        def data_gen_func1(n_batch, Tmax, seed, T0, delta):
            np.random.seed(seed)
            data = np.random.randn(n_batch, Tmax, 10)
            data[:,T0:,:] = data[:,T0:,:] + delta
            return data
        """        
        self.monitor_statistics = monitor_statistics
        self.data_gen_func0 = data_gen_func0
        self.data_gen_func1 = data_gen_func1        
        
        self.L = L
    def apply_monitoring_statistics(self, data = None,  T0 = 0, L = -1): 
        """
        Phase I analysis
        Input: data of size n_batch * Tmax * dimensions
        Output: Monitoring Statistics of size  n_batch * Tmax
        """        

        if isinstance(data, pd.DataFrame):
            data = data.values
        try:
            size = len(data[0])
        except:
            if not isinstance(data[0], (list, tuple, np.ndarray)):
                size = 1
        statistics, Tout = self.monitor_statistics(data, T0, L)
        return statistics, Tout
    
    def phase1_offline_gen(self, n_batch, Tmax, seed_list):
        """
        Phase I analysis based on generated data
        Input: 
        - n_batch: batch size for data generation
        - Tmax: generated sequence length
        - seed_list: List of seed
        
        Output: Monitoring Statistics of size  (n_batch*nseed) * Tmax
        """        

        nsim = len(seed_list)
        statistics_all = np.zeros((nsim*n_batch, Tmax))
        for i,iseed in enumerate(tqdm(seed_list)):
            data = self.data_gen_func0(n_batch, Tmax, iseed)
            statistics, Tout = self.apply_monitoring_statistics(data)
            statistics_all[(n_batch*i):(n_batch*(i+1)),:] = statistics
        return statistics_all
    
    def get_ARL(self, statisticsall, L, T0=0):
        """
        Compute ARL based on control limit L 
        Input: 
        - statistics_all: Monitoring Statistics of size  (n_batch*nseed) * Tmax
        - L: Control Limit
        Output: 
        - ARL0: the ARL0
        - truncated_percent: percentage that the ARL is truncated, if too large should increase Tmax
        - RL_all: All the Run length
        """                
        RL_all = (statisticsall[:,T0:]>L).argmax(axis=1)
        idx_truc = (statisticsall[:,T0:]>L).sum(axis = 1) ==0
        truncated_percent = np.sum(idx_truc)/len(idx_truc)
        Tmax = statisticsall.shape[1]
        RL_all[idx_truc] = Tmax
        ARL0 = np.mean(RL_all)
        return ARL0, truncated_percent, RL_all
        
    def phase1_L(self,statisticsall, r, ARL0):
        """
        Phase I analysis and use binary search to find the best control limit L
        Input: 
        - statistics_all: Monitoring Statistics of size  (n_batch*nseed) * Tmax
        - r: upperbound of the search regions
        - ARL0: the target ARL0
        Output: 
        - L: found control limit
        """        
        ARL_func = lambda L: np.mean(self.get_ARL(statisticsall,L)[0]) - ARL0        
        L = bisect(ARL_func, a=0, b=r)
        return L


    def phase1(self, n_batch, Tmax, seed_list,r,ARL0):
        """
        Complete Phase I analysis 
        Input: 
        - n_batch: batch size for data generation
        - Tmax: generated sequence length
        - seed_list: List of seed        
        - r: upperbound of the search regions
        - ARL0: the target ARL0
        Output: 
        - L: found control limit
        """        
        statisticsall = self.phase1_offline_gen(n_batch, Tmax, seed_list)
        L = self.phase1_L(statisticsall,r=r, ARL0=ARL0)
        self.L = L
        return L
        
    def test_phase1(self,n_batch, Tmax, seed_list,L):
        """
        Compute ARL0 based on control limit L 
        Input: 
        - seed_list: List of seed        
        - L: Control Limit
        Output: 
        - ARL0: the ARL0
        - truncated_percent: percentage that the ARL is truncated, if too large should increase Tmax
        - ARL_init: All the ARL0 
        """            
        statisticsall = self.phase1_offline_gen(n_batch, Tmax, seed_list)
        ARL0, truncated_percent, ARL_init = self.get_ARL(statisticsall,L)
        return ARL0, truncated_percent, ARL_init
    
    
    def phase2(self, n_batch, Tmax, seed_list, T0, delta):
        """
        Phase 2 analysis based on generated data
        Input: 
        - n_batch: batch size for data generation
        - Tmax: generated sequence length
        - seed_list: List of seed
        - T0: Start point of change
        - delta: change magnitude
        Output: Monitoring Statistics of size  (n_batch*nseed) * Tmax
        """        
        nsim = len(seed_list)
        statistics_all = np.zeros((nsim*n_batch, Tmax))
        T_all  = np.zeros(nsim*n_batch)
        for i,iseed in enumerate(tqdm(seed_list)):
            data = self.data_gen_func1(n_batch, Tmax, iseed, T0=T0, delta = delta)
            statistics, Tout = self.apply_monitoring_statistics(data, T0, self.L)
            statistics_all[(n_batch*i):(n_batch*(i+1)),:] = statistics
            T_all[(n_batch*i):(n_batch*(i+1))] = Tout - T0-1
        return statistics_all, T_all
    
