import numpy as np
import crack_detection_algorithms as CDA


data_input = np.random.randn(4000,22)
data_output = np.round(np.random.random((4000,))).astype((int))
test_input = np.random.randn(400,22)
test_output = np.round(np.random.random((400,))).astype((int))


CD_class = CDA.Parameter_Search(
                data_input,
                data_output,
            )
keys_FE = CD_class.keys_FE
keys_CA = CD_class.keys_CA
score_arr=[]

for i in keys_FE:
    for j in keys_CA:
        CD_class.FE = i
        CD_class.CDA = j
        CD_class.trained_model()
        score, predictions = CD_class.predict(test_input, test_output)
        score_arr.append(score)