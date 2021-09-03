import numpy as np
from sklearn.decomposition import KernelPCA, FastICA, PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit
import pandas as pd
from Neural_networks import autoencoder_hyperband, FCN_hyperband
import pickle
from sklearn.metrics import make_scorer, balanced_accuracy_score
from pathlib import Path
import os
### halving grid search is experimental
### if there are errors, go to sklearn\model_selection\_search_successive_halving class _SubsampleMetaSplitter:
### replace with:
# def split(self, X, y, groups=None):
#     if len(np.unique(y)) == 1:
#         raise ValueError("Only one class in training data.")
#     else:
#         while True:
#             for train_idx, test_idx in self.base_cv.split(X, y, groups):
#                 train_idx = resample(
#                     train_idx,
#                     replace=False,
#                     random_state=self.random_state,
#                     n_samples=int(self.fraction * train_idx.shape[0]),
#                 )
#                 if self.subsample_test:
#                     test_idx = resample(
#                         test_idx,
#                         replace=False,
#                         random_state=self.random_state,
#                         n_samples=int(
#                             max(1, int(self.fraction * test_idx.shape[0]))
#                         ),
#                     )
#                 yield train_idx, test_idx
#             if len(np.unique(y[train_idx])) > 1:
#                 break
class Parameter_Search:
    def __init__(self,  input_data, target_data,FE='None', CA='KNN'):
        self.FE = FE
        self.CA = CA
        self.data_input = input_data  ## scale the data!
        self.data_output = target_data.squeeze()
        self.keys_FE = [
            "ICA",
            "None",
            "K-PCA-Linear",
            "K-PCA-Sigmoid",
            "K-PCA-Polynomial",
            "K-PCA-RBF",
            #"Autoencoder",
        ]
        self.keys_CA = [
            "K-SVM-Linear",
            "K-SVM-RBF",
            "K-SVM-Polynomial",
            "K-SVM-Sigmoid",
            "KNN",
            "LR",
            #"FCN",
        ]
        self.filepath = str(Path.home()) + "\Documents\ML_comparisons"

    def trained_model(self):
        try:
            pkl_filename = (
                self.filepath + "\Models\Model_" + self.FE + "_" + self.CA + ".pkl"
            )
            with open(pkl_filename, "rb") as file:
                self.pickle_model = pickle.load(file)
        except:
            self.grid_search()

    def predict(self, Xtest, Ytest):
        score = self.pickle_model.score(Xtest, Ytest)
        predictions = self.pickle_model.predict(Xtest)
        return score, predictions

    def grid_search(self):
        # https://scikit-learn.org/stable/tutorial/statistical_inference/putting_together.html
        if os.path.isdir(self.filepath):
            pass
        else:
            os.mkdir(self.filepath)
            os.mkdir(self.filepath+'\\Models')
            os.mkdir(self.filepath + '\\Training')
            os.mkdir(self.filepath + '\\Training\\hyper_params')


        search_points = 10

        if self.FE == "None":
            FE_algorithm = No_FE()
            FE_grid = {}
            FE_grid_max = {}
            FE_grid_min = {}
        if self.FE == "K-PCA-Linear":
            points = np.arange(1, len(self.data_input[0, :]) + 1)
            FE_algorithm = PCA()
            FE_grid = {
                self.FE + "__n_components": points.astype(int),
            }
            FE_grid_max = {
                self.FE + "__n_components_maximum": np.max(points),
            }
            FE_grid_min = {
                self.FE + "__n_components_minimum": np.min(points),
            }

        if self.FE == "K-PCA-RBF":
            gamma = np.logspace(-7, 0, search_points)
            points = np.arange(1, len(self.data_input[0, :]) + 1)
            FE_algorithm = KernelPCA(kernel="rbf")
            FE_grid = {
                self.FE + "__n_components": points.astype(int),
                self.FE + "__gamma": gamma,
            }

            FE_grid_max = {
                self.FE + "__n_components_maximum": np.max(points),
                self.FE + "__gamma_maximum": np.max(gamma),
            }
            FE_grid_min = {
                self.FE + "__n_components_minimum": np.min(points),
                self.FE + "__gamma_minimum": np.min(gamma),
            }

        if self.FE == "K-PCA-Polynomial":
            degree = np.round(np.linspace(2, 3, 2))
            gamma = np.logspace(-5, 0, search_points)
            coef = np.array([0])  # np.linspace(-1,1,search_points)
            points = np.arange(1, len(self.data_input[0, :]) + 1)
            FE_algorithm = KernelPCA(kernel="rbf")
            FE_grid = {
                self.FE + "__n_components": points.astype(int),
                self.FE + "__gamma": gamma,
                self.FE + "__degree": degree.astype(int),
                self.FE + "__coef0": coef,
            }

            FE_grid_max = {
                self.FE + "__n_components_maximum": np.max(points),
                self.FE + "__gamma_maximum": np.max(gamma),
                self.FE + "__degree_maximum": np.max(degree.astype(int)),
                self.FE + "__coef0_maximum": np.max(coef),
            }
            FE_grid_min = {
                self.FE + "__n_components_minimum": np.min(points),
                self.FE + "__gamma_minimum": np.min(gamma),
                self.FE + "__degree_minimum": np.min(degree.astype(int)),
                self.FE + "__coef0_minimum": np.min(coef),
            }

        if self.FE == "K-PCA-Sigmoid":
            gamma = np.logspace(-9, 1, search_points)
            coef = np.array([0])  # np.logspace(-1, 1, search_points)
            points = np.arange(1, len(self.data_input[0, :]) + 1)
            FE_algorithm = KernelPCA(kernel="rbf")
            FE_grid = {
                self.FE + "__n_components": points.astype(int),
                self.FE + "__gamma": gamma,
                self.FE + "__coef0": coef,
            }

            FE_grid_max = {
                self.FE + "__n_components_maximum": np.max(points),
                self.FE + "__gamma_maximum": np.max(gamma),
                self.FE + "__coef0_maximum": np.max(coef),
            }
            FE_grid_min = {
                self.FE + "__n_components_minimum": np.min(points),
                self.FE + "__gamma_minimum": np.min(gamma),
                self.FE + "__coef0_minimum": np.min(coef),
            }

        if self.FE == "ICA":
            points = np.arange(1, len(self.data_input[0, :]) + 1)
            tol = np.array([1e-4])
            iterations = np.array([200])
            FE_algorithm = FastICA()
            FE_grid = {
                self.FE + "__n_components": points.astype(int),
                self.FE + "__tol": tol,
                self.FE + "__max_iter": iterations.astype(int),
            }

            FE_grid_max = {
                self.FE + "__n_components_maximum": np.max(points),
            }
            FE_grid_min = {
                self.FE + "__n_components_minimum": np.min(points),
            }

        if self.FE == "Autoencoder":
            FE_algorithm = autoencoder_hyperband()
            points = np.arange(1, len(self.data_input[0, :]) + 1)
            FE_grid = {
                self.FE + "__encoder_dim": points.astype(int),
            }

            FE_grid_max = {
                self.FE + "__n_components_maximum": np.max(points),
            }
            FE_grid_min = {
                self.FE + "__n_components_minimum": np.min(points),
            }

        #####################################################
        if self.CA == "KNN":
            points = np.round(np.linspace(1, 10, search_points))
            Class_algorithm = KNeighborsClassifier()

            CA_grid = {
                self.CA + "__n_neighbors": points.astype(int),
            }

            CA_grid_max = {
                self.CA + "__n_neighbors_maximum": np.max(points),
            }
            CA_grid_min = {
                self.CA + "__n_neighbors_minimum": np.min(points),
            }

        if self.CA == "LR":
            regularisation = np.linspace(1, 30, search_points)
            Class_algorithm = Logistic_Regression()
            threshold = np.linspace(0.2, 0.8, search_points)
            tol = np.array([1e-4])
            alg = ["lbfgs"]  # ,'newton-cg','saga']
            max_it = np.round(np.array([300]))
            CA_grid = {
                self.CA + "__Threshold": threshold,
                self.CA + "__lam": regularisation,
                self.CA + "__tol": tol,
                self.CA + "__alg": alg,
                self.CA + "__max_iter": max_it,
            }

            CA_grid_max = {
                self.CA + "__Threshold_maximum": np.max(threshold),
                self.CA + "__lam_maximum": np.max(regularisation),
                self.CA + "__tol_maximum": np.max(tol),
                self.CA + "__max+iter_maximum": np.max(max_it),
            }
            CA_grid_min = {
                self.CA + "__Threshold_minimum": np.min(threshold),
                self.CA + "__lam_minimum": np.min(regularisation),
                self.CA + "__tol_minimum": np.min(tol),
                self.CA + "__max+iter_minimum": np.min(max_it),
            }

        if self.CA == "K-SVM-Linear":
            regularisation = np.linspace(1, 30, search_points)
            Class_algorithm = SVC(kernel="linear")

            CA_grid = {
                self.CA + "__C": regularisation,
            }

            CA_grid_max = {self.CA + "__C_maximum": np.max(regularisation)}
            CA_grid_min = {self.CA + "__C_minimum": np.min(regularisation)}

        if self.CA == "K-SVM-RBF":
            regularisation = np.linspace(1, 30, search_points)
            gamma = np.logspace(-9, 2, search_points)

            Class_algorithm = SVC(kernel="linear")

            CA_grid = {
                self.CA + "__C": regularisation,
                self.CA + "__gamma": gamma,
            }

            CA_grid_max = {
                self.CA + "__C_maximum": np.max(regularisation),
                self.CA + "__gamma_maximum": np.max(gamma),
            }
            CA_grid_min = {
                self.CA + "__C_minimum": np.min(regularisation),
                self.CA + "__gamma_minimum": np.min(gamma),
            }

        if self.CA == "K-SVM-Polynomial":
            regularisation = np.linspace(1, 30, search_points)
            gamma = np.logspace(-9, 1, search_points)
            degree = np.round(np.linspace(2, 3, 2))
            coef = np.array([0])
            Class_algorithm = SVC(kernel="linear")

            CA_grid = {
                self.CA + "__C": regularisation,
                self.CA + "__gamma": gamma,
                self.CA + "__degree": degree.astype(int),
                self.CA + "__coef0": coef,
            }

            CA_grid_max = {
                self.CA + "__C_maximum": np.max(regularisation),
                self.CA + "__gamma_maximum": np.max(gamma),
                self.CA + "__degree_maximum": np.max(degree.astype(int)),
                self.CA + "__coef0_maximum": np.max(coef),
            }
            CA_grid_min = {
                self.CA + "__C_minimum": np.min(regularisation),
                self.CA + "__gamma_minimum": np.min(gamma),
                self.CA + "__degree_minimum": np.min(degree.astype(int)),
                self.CA + "__coef0_minimum": np.min(coef),
            }

        if self.CA == "K-SVM-Sigmoid":
            regularisation = np.linspace(1, 30, search_points)
            gamma = np.logspace(-9, 1, search_points)
            coef = np.array([0])
            Class_algorithm = SVC(kernel="linear")

            CA_grid = {
                self.CA + "__C": regularisation,
                self.CA + "__gamma": gamma,
                self.CA + "__coef0": coef,
            }

            CA_grid_max = {
                self.CA + "__C_maximum": np.max(regularisation),
                self.CA + "__gamma_maximum": np.max(gamma),
                self.CA + "__coef0_maximum": np.max(coef),
            }
            CA_grid_min = {
                self.CA + "__C_minimum": np.min(regularisation),
                self.CA + "__gamma_minimum": np.min(gamma),
                self.CA + "__coef0_minimum": np.min(coef),
            }

        if self.CA == "FCN":

            Threshold = np.linspace(0, 1, search_points)
            layers = [1, 2, 3]
            regularisation = np.logspace(-2, 1, search_points)
            units = [32, 64, 128]
            learning_rate = np.linspace(1e-1, 1, search_points)
            batches = [1, 8, 16]
            search_algorithm = ["Adam"]  # , 'RMSprop', 'GD']
            epochs = [100]

            Class_algorithm = FCN_hyperband()

            CA_grid = {
                self.CA + "__Threshold": Threshold,
                self.CA + "__epochs": [5],
                self.CA + "__factor": [3],
                self.CA + "__patience": [3],
            }
            CA_grid_max = {
                self.CA + "__Threshold_maximum": np.max(Threshold),
            }
            CA_grid_min = {
                self.CA + "__Threshold_minimum": np.min(Threshold),
            }

        pipe = Pipeline(steps=[(self.FE, FE_algorithm), (self.CA, Class_algorithm)])

        # pipe.get_params().keys()

        param_grid = {**FE_grid, **CA_grid}  # concat dicts

        # scorer = make_scorer(f1_score)
        scorer = make_scorer(
            balanced_accuracy_score
        )  # https://scikit-learn.org/stable/modules/model_evaluation.html#balanced-accuracy-score
        try:
            search = HalvingGridSearchCV(
                pipe,
                param_grid,
                cv=ShuffleSplit(test_size=0.20, n_splits=1, random_state=0),
                n_jobs=-1,
                verbose=3,
                scoring=scorer,
                min_resources=500,
            )
            search.fit(self.data_input, self.data_output)

            if self.CA == "FCN":
                param_grid = search.best_params_
                for i in search.best_params_:
                    x = search.best_params_.get(i)
                    param_grid[i] = [x]

                param_grid["FCN__epochs"] = [100]
                param_grid["FCN__factor"] = [3]
                param_grid["FCN__patience"] = [10]

                pipe = Pipeline(
                    steps=[(self.FE, FE_algorithm), (self.CA, Class_algorithm)]
                )

                search = HalvingGridSearchCV(
                    pipe,
                    param_grid,
                    cv=ShuffleSplit(test_size=0.20, n_splits=1, random_state=0),
                    n_jobs=-1,
                    verbose=3,
                    scoring=scorer,
                    min_resources=500,
                )

                search.fit(self.data_input, self.data_output)

        except:

            search = GridSearchCV(
                pipe,
                param_grid,
                cv=ShuffleSplit(test_size=0.20, n_splits=1, random_state=0),
                n_jobs=-1,
                verbose=3,
                scoring=scorer,
            )

            search.fit(self.data_input, self.data_output)

            if self.CA == "FCN":
                param_grid = search.best_params_
                for i in search.best_params_:
                    x = search.best_params_.get(i)
                    param_grid[i] = [x]

                param_grid["FCN__epochs"] = [100]
                param_grid["FCN__factor"] = [3]
                param_grid["FCN__patience"] = [10]

                pipe = Pipeline(
                    steps=[(self.FE, FE_algorithm), (self.CA, Class_algorithm)]
                )

                search = GridSearchCV(
                    pipe,
                    param_grid,
                    cv=ShuffleSplit(test_size=0.20, n_splits=1, random_state=0),
                    n_jobs=-1,
                    verbose=3,
                    scoring=scorer,
                )

                search.fit(self.data_input, self.data_output)

        best_params = pd.DataFrame(search.best_params_, index=[0])
        print("Best parameter (CV score=%0.3f):" % search.best_score_)
        print(search.best_params_)
        file = self.filepath + "\\Training\hyper_params\\" + self.FE + "_" + self.CA + ".xlsx"

        check = []
        for i, j in search.best_params_.items():
            if j == CA_grid_min.get(i + "_minimum"):
                check.append("At limit")
            elif j == CA_grid_max.get(i + "_maximum"):
                check.append("At limit")
            elif j == FE_grid_min.get(i + "_minimum"):
                check.append("At limit")
            elif j == FE_grid_max.get(i + "_maximum"):
                check.append("At limit")
            else:
                check.append("good")

        check = pd.DataFrame(check).transpose()
        check.columns = list(best_params.columns)
        best_params = best_params.append(check, ignore_index=True)
        best_params.to_excel(file, index=False)

        try:
            df_all = pd.read_excel(
                self.filepath + "\\Training\\algorithm_performance.xlsx"
            )
            df_current = pd.DataFrame(
                [self.FE, self.CA, search.best_score_]
            ).transpose()
            df_current.columns = ["FE", "CA", "Accuracy"]
            df_all = df_all.append(df_current, ignore_index=True)

        except:
            df_all = pd.DataFrame([self.FE, self.CA, search.best_score_]).transpose()

        df_all.columns = ["FE", "CA", "Accuracy"]
        df_all.to_excel(
            self.filepath + "\\Training\\algorithm_performance.xlsx", index=False
        )

        if not self.CA == "FCN" and not self.FE == "Autoencoder":
            pkl_filename = (
                self.filepath + "\Models\Model_" + self.FE + "_" + self.CA + ".pkl"
            )

            with open(pkl_filename, "wb") as file:
                pickle.dump(search.best_estimator_, file)

        self.pickle_model = search.best_estimator_


class No_FE(BaseEstimator, TransformerMixin):
    def fit(self, X, y):  # load encoder if it exists! for current classifier
        return self

    def transform(self, data, y=None):
        return data


class Logistic_Regression(BaseEstimator, ClassifierMixin):
    def __init__(self, lam=1, Threshold=0.5, tol=1e-4, alg="lbfgs", max_iter=200):
        self.lam = lam
        self.Threshold = Threshold
        self.tol = tol
        self.alg = alg
        self.max_iter = max_iter

    def fit(self, X, y):  # load encoder if it exists! for current classifier
        self.LR = LogisticRegression(
            C=self.lam, solver=self.alg, tol=self.tol, max_iter=self.max_iter
        )

        self.input_data = X
        self.output_data = y

        self.LR.fit(self.input_data, self.output_data)
        P = self.LR.predict_proba(self.input_data)

        # self.LR.classes_ # to see which label is which

        predicted = np.ones((len(P), 1)) * np.nan
        P_crack = P[:, 1]
        for pp in range(0, len(P_crack.squeeze())):
            if P_crack[pp] > self.Threshold:
                predicted[pp] = 1
            else:
                predicted[pp] = 0

        return predicted.squeeze()

    def predict(self, X):

        P = self.LR.predict_proba(X)

        predicted = np.ones((len(P), 1)) * np.nan
        P_crack = P[:, 1]
        for pp in range(0, len(P_crack.squeeze())):
            if P_crack[pp] > self.Threshold:
                predicted[pp] = 1
            else:
                predicted[pp] = 0

        return predicted.squeeze()


# class Logistic_Regression(BaseEstimator, ClassifierMixin):
#     def __init__(self, lam=0,Threshold=0.5):
#         self.lam = lam
#         self.Threshold = Threshold
#
#     def calc_probabilities(self,X,B):
#         Z = np.matmul(X, B)
#         P = np.divide(1, 1 + np.exp(-Z))
#         return Z,P
#
#     def define_matrix(self,input_data):
#         num_samples = len(input_data[:, 0])
#         X = np.concatenate((np.ones((num_samples, 1)), input_data), axis=1)
#         return X
#
#     def logistic_regression_coefficients(self):
#
#         X = self.define_matrix(self.input_data)
#         B = np.ones((len(X[0, :]), 1))
#
#         fun = np.ones((100,))*np.nan
#
#         for kk in range(0,100):
#
#             Z,P = self.calc_probabilities(X,B)
#             XX = np.ones((len(Z.squeeze()), 1)) * np.nan
#             H = np.ones((len(B.squeeze()), len(B.squeeze()))) * np.nan
#
#             fun[kk] = -np.sum(np.multiply(np.log(P),self.output_data) + np.multiply((1 - self.output_data), np.log(1 - P)))
#
#             D = -(self.output_data.squeeze() - P.squeeze())
#             D = D.reshape((len(D),1))
#             gradient = np.transpose(np.matmul(np.transpose(D),X))
#
#             for k in range(0,len(B.squeeze())):
#                 for j in range(0,len(B.squeeze())):
#                     for N in range(0,len(Z.squeeze())):
#                         XX[N, 0] = X[N, k] * X[N, j]
#
#                     H[k, j] = -np.sum(np.multiply(np.multiply(XX, P),P))
#
#
#
#             B = B + np.matmul(np.linalg.inv(H), (gradient))
#
#         return B
#
#
#     def fit(self,X,y):  #load encoder if it exists! for current classifier
#         self.input_data = X
#         self.output_data = y
#
#         X = self.define_matrix(X)
#         self.B = self.logistic_regression_coefficients()
#         Z,P = self.calc_probabilities(X,self.B)
#
#         predicted = np.ones((len(P),1))*np.nan
#         for pp in range(0,len(P.squeeze())):
#             if P[pp,0] > self.Threshold:
#                 predicted[pp] = 1
#             else:
#                 predicted[pp] = 0
#
#         return predicted
#
#     def predict(self,X):
#
#         X = self.define_matrix(X)
#         Z, P = self.calc_probabilities(X, self.B)
#
#         predicted = np.ones((len(P), 1)) * np.nan
#         for pp in range(0, len(P.squeeze())):
#             if P[pp, 0] > self.Threshold:
#                 predicted[pp] = 1
#             else:
#                 predicted[pp] = 0
#
#         return predicted
