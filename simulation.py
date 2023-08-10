import os
import scipy
import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
# from ml.vis import cal_metric, model_efficiency
from eval.eval_method import fcv, cv, forward_holdout, forward_holdout_split, cal_metric, model_efficiency, \
    score_dp_by_forward_holdout
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")




def dict_append(d, key, obj):
    if key in list(d.keys()):
        d[key].append(obj)
    else:
        d[key] = []
        d[key].append(obj)


if __name__ == '__main__':
    for eval_score in ['DP', 'RMSE']:

        # material benchmark datasets for regression testing
        datasets_info = [
            # {"dataset_name": 'steel_strength', 'target_col': "tensile strength"},
            {"dataset_name": 'brgoch_superhard_training', 'target_col': "shear_modulus"},
            {"dataset_name": 'double_perovskites_gap', 'target_col': 'gap gllbsc'},
            {"dataset_name": 'superconductivity2018', 'target_col': 'Tc'},
            {"dataset_name": 'matbench_expt_gap', 'target_col': 'gap expt'},
            {"dataset_name": 'castelli_perovskites', 'target_col': 'e_form'},
            {"dataset_name": 'expt_formation_enthalpy', 'target_col': "e_form expt"},
            {"dataset_name": 'expt_formation_enthalpy_kingsbury', 'target_col': "expt_form_e"},
            {"dataset_name": 'expt_gap', 'target_col': "gap expt"},
            {"dataset_name": 'wolverton_oxides', 'target_col': "e_form"},
        ]

        val_config = {
            "feature_num": 20,
            "cv_fold": 5,
            'test_ratio': 0.10  # extra test data ratio
        }
        method_config = {
            'CV': {"metric": ['R2', 'RMSE', 'MAE']},
            'FCV': {"metric": ['R2', 'RMSE', 'MAE', 'DP']},
            'FH': {"metric": ['R2', 'RMSE', 'MAE', 'DP']}
        }
        # record all result
        result_df = pd.DataFrame()
        Y_col = 'target'

        test_ratio = val_config['test_ratio']

        # alg
        alg_dict = {
            # "Lasso": Lasso(),
            # "Ridge": Ridge(),
            # "LinearRegression": LinearRegression(),
            'LinearSVR': SVR(kernel='linear'),
            "GradientBoosting": GradientBoostingRegressor(random_state=0),
            "ExtraTrees": ExtraTreesRegressor(random_state=0),
            "RandomForest": RandomForestRegressor(random_state=0),
            "KNeighbors": KNeighborsRegressor(),
            "DecisionTree": DecisionTreeRegressor(random_state=0),
            'RbfSVR': SVR(kernel='rbf')
        }
        for info in datasets_info:  # dataset iteration
            # region read dataset and preprocessing
            dataset_name = info['dataset_name']
            target_col = info['target_col']
            data_file_name = f"./data/{dataset_name}_{target_col}.csv"  # save dataset with calculated features
            print(f"dataset_name:{dataset_name}")

            if os.path.exists(data_file_name):
                ml_dataset = pd.read_csv(data_file_name)
                features = list(ml_dataset.columns[:-1])  # target is the last column
            else:
                raise Exception(f"{data_file_name} dataset not find")
            ml_dataset = ml_dataset.sort_values(by=Y_col, ascending=True)
            X = ml_dataset[features]
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
            Y = ml_dataset[Y_col].values
            sc = StandardScaler()
            Y = sc.fit_transform(np.array(Y).reshape(-1, 1)).flatten()
            # print(Y)
            print(f"samples counts :", len(Y))
            print(f"Y :{target_col} from {min(Y)}~{max(Y)}")
            # delete constant features
            from sklearn.feature_selection import VarianceThreshold

            var = VarianceThreshold(threshold=0)
            X = var.fit_transform(X)
            from sklearn.feature_selection import SelectKBest
            from sklearn.feature_selection import f_regression
            from sklearn.preprocessing import StandardScaler

            # endregion
            # init training set
            n_rows = len(X)
            all_index = list(range(n_rows))
            # init
            random.seed(0)
            np.random.seed(0)
            train_index = random.sample(list(range(round(0.8 * n_rows))), round(0.2 * n_rows))
            top_n = round(0.01 * n_rows)
            # if find Y > Y_thr then
            Y_thr = Y[round(-0.2 * n_rows)]
            score_list = []  # trace the best score in iteration
            iter_n = 0  # total iter
            dp_score_list = []  # trace the dp score in iteration
            me_score_list = []
            for i in range(100):
                iter_n = i
                X_train = X[train_index, :]
                Y_train = Y[train_index]
                # feature selection
                feature_selection = SelectKBest(f_regression, k=20).fit(X_train, Y_train)
                X_train = feature_selection.transform(X_train)
                material_space = list(set(list(all_index)) - set(train_index))
                # region of model fit and eval
                best_score = 10 ** 10
                best_model = None
                best_model_dp_score = None
                for alg_name in alg_dict.keys():
                    model = alg_dict[alg_name]
                    model.fit(X_train, Y_train)
                    # lower score is better
                    dp_score = -score_dp_by_forward_holdout(model, X_train, Y_train,
                                                            test_ratio=0.10, reverse=True, cv_fold=5,
                                                            )
                    y_true, y_predict, X_train_h, y_train_h = forward_holdout(model, X_train, Y_train,
                                                                              test_ratio=0.25,
                                                                              reverse=True)
                    evaluation_matrix = cal_metric(y_true, y_predict)
                    evaluation_matrix['DP'] = dp_score
                    metrics = ['R2', 'R', 'RMSE', 'MAE']
                    # set eval score for model selection
                    score = evaluation_matrix[eval_score]
                    if score < best_score:
                        best_score = score
                        best_model = model
                        best_model_dp_score = dp_score
                print("best model", best_model, "best score", best_score)

                score_list.append(best_score)
                dp_score_list.append(best_model_dp_score)

                X_exp = feature_selection.transform(X[material_space, :])
                y_predict = best_model.predict(X_exp)
                # find index with max prediction
                max_index = np.argsort(-y_predict)[:top_n]
                # update train index and material space
                train_index = train_index + list(np.array(material_space)[max_index])
                material_space = list(set(list(all_index)) - set(train_index))
                print(f"iteration{i}")
                print("train_size", len(train_index))
                print("material_space_size", len(material_space))
                y_target = [y for y in Y[train_index] if y >= Y_thr]
                print("find material num:", len(y_target))

                if len(y_target) >= 0.10 * n_rows:
                    break
            print(f"mean score{np.mean(score_list):2f}")
            print(f"mean dp score{np.mean(dp_score_list):2f}")
            print(f"std score{np.std(score_list):2f}")
            print(f"std dp score{np.std(dp_score_list):2f}")
            print(f"total iteration: {iter_n + 1}")
            result_one = {"dataset": [dataset_name],
                          "score mean": [np.mean(score_list)],
                          "DP mean": [np.mean(dp_score_list)],
                          "score std": [np.std(score_list)],
                          "DP std": [np.std(dp_score_list)],
                          "total iteration": [iter_n + 1],
                          "find material": [len(y_target)/n_rows]}
            res_df_one = pd.DataFrame(result_one)  # result of one dataset
            result_df = pd.concat([result_df, res_df_one], axis=0)
        result_df.to_csv(f'simulation_{eval_score}.csv', index=False)
