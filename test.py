import numpy as np
import pandas as pd

import crack_detection_algorithms as CDA

class testing():
    def test_multiple_input(self):
        data_input = np.random.randn(3000,5)
        data_output = np.round(np.random.random((4000,))).astype((int))
        test_input = np.random.randn(300,5)
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


    def test_single_input(self):
        data_input = np.random.randn(3000, 1)
        ind = np.where(data_input < 0)
        data_output = np.zeros(np.shape(data_input))
        data_input = data_input + np.random.randn(3000, 1)*0.2 # add noise

        data_output[ind] = 1

        test_input = np.random.randn(300, 1)
        ind = np.where(test_input < 0)
        test_output = np.zeros(np.shape(test_input))
        test_input = test_input + np.random.randn(300, 1) * 0.2  # add noise
        test_output[ind] = 1

        CD_class = CDA.Parameter_Search(
            data_input,
            data_output,
        )
        keys_FE = CD_class.keys_FE
        keys_CA = CD_class.keys_CA
        score_arr = []

        for i in keys_FE:
            for j in keys_CA:
                CD_class.FE = i
                CD_class.CA = j
                CD_class.trained_model()
                score, predictions = CD_class.predict(test_input, test_output)
                score_arr.append(score)

                filepath = CD_class.filepath
                df = pd.DataFrame([keys_FE,keys_CA,score_arr]).transpose()
                df.columns = ["FE", "CA", "Accuracy"]
                df.to_excel(
                    filepath + "\\algorithm_performance_test.xlsx", index=False
                )





T = testing()
T.test_single_input()