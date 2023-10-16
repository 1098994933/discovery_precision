import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from eval.eval_method import fcv, cv, forward_holdout, forward_holdout_split, cal_metric, discovery_precision


def dict_append(d, key, obj):
    if key in list(d.keys()):
        d[key].append(obj)
    else:
        d[key] = []
        d[key].append(obj)


if __name__ == '__main__':
    # material benchmark datasets for regression testing
    datasets_info = [
        {"dataset_name": 'steel_strength', 'target_col': "tensile strength"},
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
        'FH': {"metric": ['R2', 'RMSE', 'MAE']}
    }
    # record all result
    result_df = pd.DataFrame()
    Y_col = 'target'

    test_ratio = val_config['test_ratio']

    # alg
    alg_dict = {
        "Lasso": Lasso(),
        "Ridge": Ridge(),
        "LinearRegression": LinearRegression(),
        'LinearSVR': SVR(kernel='linear'),
        "GradientBoosting": GradientBoostingRegressor(),
        "ExtraTrees": ExtraTreesRegressor(),
        "RandomForest": RandomForestRegressor(),
        "KNeighbors": KNeighborsRegressor(),
        "DecisionTree": DecisionTreeRegressor(),
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

        X = ml_dataset[features]
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        Y = ml_dataset[Y_col]
        print(f"samples counts :", len(Y))
        print(f"Y :{target_col} from {min(Y)}~{max(Y)}")
        # delete constant features
        from sklearn.feature_selection import VarianceThreshold

        var = VarianceThreshold(threshold=0)
        X = var.fit_transform(X)

        # endregion
        for reverse in [True, False]:
            if reverse:  # higher means better
                Y = Y
            else:
                Y = -Y
            result_info = {}  # result for one task
            # split all data into train and test, False for higher extra task
            X_train, Y_train, X_test, Y_test = forward_holdout_split(X, Y, test_ratio, reverse=reverse)
            # check dataset
            Y_test_max = max(Y_test)
            Y_test_min = min(Y_test)
            if Y_test_max == Y_test_min:
                print('skip the dataset for Y_test has no variance')
                continue

            from sklearn.feature_selection import SelectKBest
            from sklearn.feature_selection import f_regression
            from sklearn.preprocessing import StandardScaler

            feature_selection = SelectKBest(f_regression, k=val_config['feature_num']).fit(X_train, Y_train)
            X_train = feature_selection.transform(X_train)
            X_test = feature_selection.transform(X_test)
            sc = StandardScaler()
            Y_train = sc.fit_transform(np.array(Y_train).reshape(-1, 1)).flatten()
            Y_test = sc.transform(np.array(Y_test).reshape(-1, 1)).flatten()

            # region ML modeling and eval
            for alg_name in alg_dict.keys():
                model = alg_dict[alg_name]

                val_set = 'val'
                # region cv
                val_method = 'CV'
                metrics = method_config[val_method]['metric']
                y_true, y_predict = cv(model, X_train, Y_train, k=val_config['cv_fold'])
                cv_predict = y_predict.copy()
                evaluation_matrix = cal_metric(y_true, y_predict)
                cv_r2 = evaluation_matrix['R2']
                if cv_r2 < 0.3:
                    print(f"skip the model R2={cv_r2}")
                    continue

                dict_append(result_info, 'dataset_name', dataset_name)
                dict_append(result_info, 'target_col', target_col)
                dict_append(result_info, 'reverse', reverse)
                dict_append(result_info, 'alg_name', alg_name)

                for metric in metrics:
                    dict_append(result_info, f'{val_method} {metric} {val_set}', evaluation_matrix[metric])

                # endregion

                # region fcv
                val_method = 'FCV'
                y_true, y_predict, details = fcv(model, X_train, Y_train, k=5, details=True)
                dp_scores = [discovery_precision(details['y_test_predicts'][i], details['y_train_predicts'][i],
                                                 details['y_tests'][i], details['y_trains'][i])
                             for i in range(len(details['y_trains']))]
                score = np.median(dp_scores)

                r2_scores = [r2_score(details['y_test_predicts'][i], details['y_tests'][i]) for i in
                             range(len(details['y_trains']))]
                r2 = np.median(r2_scores)

                evaluation_matrix = cal_metric(y_true, y_predict)
                evaluation_matrix['R2'] = r2
                metrics = ['R2', 'R', 'RMSE', 'MAE']
                for metric in metrics:
                    dict_append(result_info, f'{val_method} {metric} {val_set}', evaluation_matrix[metric])
                metric = 'DP'
                dict_append(result_info, f'{val_method} {metric} {val_set}', score)
                # endregion

                # region forward holdout
                val_method = 'FH'
                y_true, y_predict, X_train_h, y_train_h = forward_holdout(model, X_train, Y_train,
                                                                          test_ratio=test_ratio,
                                                                          reverse=reverse)
                evaluation_matrix = cal_metric(y_true, y_predict)
                metrics = ['R2', 'R', 'RMSE', 'MAE']
                for metric in metrics:
                    dict_append(result_info, f'{val_method} {metric} {val_set}', evaluation_matrix[metric])
                model.fit(X_train_h, y_train_h)
                # training data predictions
                y_train_predict = model.predict(X_train_h)
                # get cv prediction
                y_inter_true, y_train_predict = cv(model, X_train_h, y_train_h, k=val_config['cv_fold'])

                y_train_predict_valid = y_train_predict.copy()
                y_predict_valid = y_predict.copy()

                valid_me = score

                y_extra_predict = y_predict
                y_inter_predict = y_train_predict
                y_extra_true = y_true
                y_inter_true = y_inter_true

                score = discovery_precision(y_extra_predict, y_inter_predict, y_extra_true, y_inter_true)
                metric = f'DP'
                dict_append(result_info, f'{val_method} {metric} {val_set}', score)
                method_config['FH']["metric"].append(metric)
                # endregion

                model_val = alg_dict[alg_name]
                print('=' * 60)
                print(model_val)
                # region model testing
                # ====================== record performance start ========================
                val_set = 'test'
                model_val.fit(X_train, Y_train)
                y_predict = model_val.predict(X_test)

                y_true = Y_test
                y_train_predict = model_val.predict(X_train)

                # cv to get out of bag prediction
                y_inter_true, y_train_predict = cv(model_val, X_train, Y_train, k=val_config['cv_fold'])
                y_extra_predict = y_predict
                y_inter_predict = y_train_predict
                y_extra_true = y_true
                y_inter_true = y_inter_true

                test_me = score
                evaluation_matrix = cal_metric(y_true, y_predict)
                # ====================== record performance========================

                for metric in metrics:
                    dict_append(result_info, f'{metric} {val_set}', evaluation_matrix[metric])

                score = discovery_precision(y_extra_predict, y_inter_predict, y_extra_true, y_inter_true)
                metric = f'DP'
                dict_append(result_info, f'{metric} {val_set}', score)
                # endregion

                # ====================== record performance end ========================
                # result for one data set
                print(result_info)
                res_df_one = pd.DataFrame(result_info)  # result of one dataset
            if len(res_df_one) > 2:
                print(res_df_one)
                # cal rank of test error
                metric_test = method_config['FH']["metric"]
                for metric in metric_test:
                    test_col = f'rank_{metric}_test'
                    res_df_one[test_col] = res_df_one[f'{metric} test'].rank(method='first', ascending=False)

                # cal rank of val error with eval methods
                for method in method_config.keys():
                    for metric in method_config[method]["metric"]:
                        val_col = f'rank_{method}_{metric}_val'
                        res_df_one[val_col] = res_df_one[f'{method} {metric} val'].rank(method='first',
                                                                                        ascending=False)
                result_df = pd.concat([result_df, res_df_one], axis=0)
            # endregion
        # summary the result clean the result_info
        result_df.to_csv('result_df.csv')
    print("finished")
