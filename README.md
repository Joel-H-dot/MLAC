Please reference this package as:

Hampton, J.; Tesfalem; H., Fletcher ; A.,Williams, K ; Peyton, A. ; Brown, M (2021). 
A Comparison of Machine Learning Algorithms for Notch Classification in a Conductive Block in
the Presence of Electrical Conductivity Variations.

# Overview

MLAC is [available on PyPI][pypi], and can be installed via
```none
pip install MLAC
```
This package provides the functionality to quickly compare seventypes of feature extraction algorithms and seven types of classifiers. In total there are 49 unique algorithms which can be defined from these FE and classifier algorithms, using the Sci-Kit learn pipeline and grid search functions. Included are two neural networks: an auto-encoder and vanilla fully connected network. The papermeters in these are found using the HyperBand algorithm, provided in the KErasTuner package. When a neural network is included in the pipeline, an initial search is perfomed over the hyper-parameter space using a low number of epochs, patience and factor; this is increased at the end of the search to provide a finer search. This process faciliates quickly determinng a good set of hyper-parameters. Included is the ability to see the different hyper-parameter values selected, and whether these are at the bounds of the defined search.  


[pypi]:  https://pypi.org/project/MVPR/

# Example
Import data:
```
data_input = np.random.randn(3000, 1)
ind = np.where(data_input < 0)
data_output = np.zeros(np.shape(data_input))
data_input = data_input + np.random.randn(3000, 1)*0.2 # add noise
data_output[ind] = 1

test_input = np.random.randn(300, 1)
ind = np.where(test_input < 0)
test_output = np.zeros(np.shape(test_input))
test_input = test_input + np.random.randn(300, 1)* 0.2 # add noise
test_output[ind] = 1
```
This looks like:
![data](https://user-images.githubusercontent.com/60707891/131995760-2e2734ca-161b-4482-b758-f4c4d03c8858.png)

We want to find some mapping function for the same input data. Using the MVPR code we can place the vectors into a matrix as (1). This matrix of target data can be split into training, validation or test and passed directly into an MVPR class object. Alternatively, they can be passed seperately into two different instantiations, such that different polynomial orders can be used. 

![image](https://user-images.githubusercontent.com/60707891/115009673-70d89780-9ea4-11eb-97f3-a02e29d4fb30.png)


First import the data:
```
import MVPR as MVP
import numpy as np
import pandas as pd
from openpyxl import load_workbook
df= pd.read_excel(r'C:\Users\filepath\data.xlsx')
data=df.to_numpy()
df= pd.read_excel(r'C:\Users\filepath\targets.xlsx')
targets=df.to_numpy()
```
select the proportions of data for cross-validation
```
proportion_training = 0.9
num_train_samples = round(len(data[:,0])*proportion_training)
num_val_samples = round(len(data[:,0]))-num_train_samples
```
standardise:
```
mean_dat = data[:, :].mean(axis=0)
std_dat = data[:, :].std(axis=0)

data -= mean_dat

if 0 not in std_dat:
    data[:, :] /= std_dat

training_data = data[:num_train_samples, :]
training_targets = targets[:num_train_samples, :]

validation_data = data[-num_val_samples :, :]
validation_targets = targets[-num_val_samples :, :]
```
call the following
```
M = MVP.MVPR_forward(training_data, training_targets, validation_data, validation_targets)

optimum_order = M.find_order()
coefficient_matrix = M.compute_CM(optimum_order)

predicted_validation = M.compute(coefficient_matrix, optimum_order, validation_data)

df = pd.DataFrame(predicted_validation)
df.to_excel(r'C:\Users\filepath\predicted.xlsx')
```
The fitted curves:

![image](https://user-images.githubusercontent.com/60707891/115009854-a5e4ea00-9ea4-11eb-8774-6c87cf89c7b5.png)

![image](https://user-images.githubusercontent.com/60707891/115009871-abdacb00-9ea4-11eb-9d12-b76d45b67835.png)

# Functions and arguments
```
import MVPR as MVP
:
MVPR_object = MVP.MVPR_forward(training_data, training_targets, validation_data, validation_targets, verbose=True, search = 'exponent')
```
The verbose argument is optional, its default value is false. 

The truncation point can either be written as 10^a or as b, the input argument 'search' specifies whether we search for the value of a or b; see the following code segment from the golden section search: 
```
:
if self.search == 'exponent':
   distance = 0.61803398875 * (np.log10(ind_high_1) - np.log10(ind_low_1))
   ind_low_2 = round(10 ** (np.log10(ind_high_1) - distance))
   ind_high_2 = round(10 ** (np.log10(ind_low_1) + distance))
else:
   distance = 0.61803398875 * (ind_high_1 - ind_low_1)
   ind_low_2 = round(ind_high_1 - distance)
   ind_high_2 = round(ind_low_1 + distance)
:
```
___________________________________________________________________________________________________________
```
optimal_order=MVPR_object.find_order()
```
This function finds the optimal order of polynomial in the range 0 to 6, using cross validation. 
___________________________________________________________________________________________________________
```
MVPR_object.compute_CM(order)
```
This function computes the coefficient matrix which fits a polynomial to the measured data in a least squares sense. The fit is regularised using truncated singular value decomposition, which eliminates singular values under a certain threshold. Any oder can be passed into this by the user, it does not have to have the range limited in find_oder(). 

# Theory 

 For the theory behind the code see [[1]](#1) and [[2]](#2). 

## References
<a id="1">[1]</a> 
Hansen, P. C.  (1997). 
Rank-deficient and Discrete Ill-posed Problems: Numerical Aspects of Linear Inversion. 

<a id="2">[2]</a> 
Hampton, J., Tesfalem, H., Fletcher, A., Peyton, A., Brown, M (2021). 
Reconstructing the conductivity profile of a graphite block using inductance spectroscopy with data-driven techniques. 
Insight - Non-Destructive Testing and Condition Monitoring, 63(2), 82-87.

