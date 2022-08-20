# Readme

This is the code for the paper "Adaptive Partially-Observed Sequential Change Detection and Isolation".
The paper is accepted in Technometrics and the Arxiv version is provided [here](https://arxiv.org/abs/2208.08855).

## Code
1. [srpabstract.py](model/srpabstract.py): A class that do SRP statistics

- Initialization: 
  - p: Number of dimensions
  - c: scale vector, Target meanshift is c * M
  - k: Number of failuer Mode
  - M: Failure Mode Mean Matrix of k failure modes: p * k 
  - nsensors: number of selected sensors     
  - Ks: Number of selected failure mode
  - L: control limit, set to -1 if not initialized yet.

You need to override the following functions

- `compute_log_LRT`: Compute the log liklihood ratio, 
- `compute_index`: Compute the index function to decide the best sensing allocation

Other implemented method: 

- `compute_monitoring_statistics`: Computed SRP statistics using the Top-R rules
- `compute_monitor_batch`: Compute the monitoring results for batch of samples

2. [TSSRP.py](model/TSSRP.py): Implement original TSSRP method

- srpabstract initialization except 
  - mode: which testing statistics to use, Default to `T2`

- compute_index: Sensor index becomes the failure mode index



3. [ExtendedTSSRP.py](model/ExtendedTSSRP.py)

- srpabstract initialization except 
  - mode: which testing statistics to use, Default to `T2`

- compute_index
  - Mode `T2` Default, using summation of log SRP.
  - Mode `T1` , using summation of  SRP, no closed forms and using greedy algorithm
  - Mode `T1_Max` , using summation of  SRP, no closed formsusing Max approximation

4. [spc.py](spc/spc.py): A generic class for process monitoring using simulation 

- Initialization: 
  - `monitor_statistics`: given input of n_batch * Tmax * dimensions, return the monitoring statistics of size n_batch * Tmax denote
  - `data_gen_func0`: Generate normal samples, return n_batch * Tmax * dimensions
  - `data_gen_func1`: Generate abnormal samples, return n_batch * Tmax * dimensions
- `phase1`: 
  - Generate iterations (= number of seeds) of each data with n_batch size
  - Use Binary search to find the control limit `L` where the `ARL0` is fixed

- `phase2`:
  - Generate iterations (= number of seeds) of each data with n_batch size
  - Return ARL1 
  
  

## Dataset
There are three datasets that used in the paper: 
1. The dataset is offered in [temperature.mat](data/temperature.mat). Thermal imaging monitoring of 3D printing: The dataset used is publically available at [figshare Scenario 3](https://figshare.com/articles/dataset/DATASET_from_Spatially_weighted_PCA_for_monitoring_video_image_data_with_application_to_additive_manufacturing_by_B_M_Colosimo_and_M_Grasso_JQT_2018/7092863). 
data
2. Tonnage dataset, which is offered in [tonnage.mat](data/tonnage.mat).
2. COVID-19 monitoring of different counties in WA. The dataset is derived from the [JHU](https://github.com/CSSEGISandData/COVID-19). The WA dataset used in the paper has been also stored in [Infection rate](data/Infection_Proportion.csv). 


