
# Overview

MLAC is [available on PyPI][pypi], and can be installed via
```none
pip install MLAC
```
This package provides the functionality to quickly compare seven types of feature extraction algorithms and seven types of classifiers. In total there are 49 unique algorithms which can be defined from these FE and classifier algorithms, using the Sci-Kit learn pipeline and grid search functions. Included are two neural networks: an auto-encoder and vanilla fully connected network. The parameters in these are found using the HyperBand algorithm, provided in the KerasTuner package. When a neural network is included in the pipeline, an initial search is performed over the hyper-parameter space using a low number of epochs, patience and factor; this is increased at the end of the search to provide a finer search. This process facilitates quickly determining a good set of hyper-parameters. Included is the ability to see the different hyper-parameter values selected, and whether these are at the bounds of the defined search.    


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

We want to find the best ML algorithm that can seperate out this data and classify it:

```
CD_class = CDA.Parameter_Search(
            data_input,
            data_output,
        )
```
To see the available algorithms:
```
keys_FE = CD_class.keys_FE
keys_CA = CD_class.keys_CA
```
Finally:
```
score_arr = []
for i in keys_FE:
   for j in keys_CA:
       CD_class.FE = i
       CD_class.CA = j
       CD_class.trained_model()
       score, predictions = CD_class.predict(test_input, test_output)
       score_arr.append(score)
```
The results are save automatically for the training data:

![image](https://user-images.githubusercontent.com/60707891/131996611-4167bdc2-4533-401e-aa17-cce9bcb5ac66.png)

We can see the results of a specific hyper-parameter search:

![image](https://user-images.githubusercontent.com/60707891/131996887-3e7db117-4453-443d-9ae6-8427adb3f1de.png)

If you want to save the data, you can pull the default filepath:

```
filepath = CD_class.filepath
df = pd.DataFrame([keys_FE,keys_CA,score_arr]).transpose()
df.columns = ["FE", "CA", "Accuracy"]
df.to_excel(filepath + "\\algorithm_performance_test.xlsx", index=False)
```

see:

![image](https://user-images.githubusercontent.com/60707891/131998255-f9143ba2-0af8-4385-8715-6d66a8aca74c.png)

