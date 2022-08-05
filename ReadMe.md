# Readme
## Dataset
There are four datasets that used in the paper: 
1. Solar Flare Dataset outbreak video frames
2. COVID-19 monitoring of different counties in WA
3. Thermal imaging monitoring of 3D printing
## Code
1. srpabstract.py: A class that do SRP statistics

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

2. TSSRP.py: Implement original TSSRP method

- srpabstract initialization except 
  - mode: which testing statistics to use, Default to `T2`

- compute_index: Sensor index becomes the failure mode index



3. ExtendedTSSRP.py

- srpabstract initialization except 
  - mode: which testing statistics to use, Default to `T2`

- compute_index
  - Mode `T2` Default, using summation of log SRP.
  - Mode `T1` , using summation of  SRP, no closed forms and using greedy algorithm
  - Mode `T1_Max` , using summation of  SRP, no closed formsusing Max approximation

4. spc.py: A generic class for process monitoring using simulation 

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
