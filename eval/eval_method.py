import pandas as pd
import math
import sys
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np
from scipy.stats import pearsonr


def fcv(original_model, X, y, minimum_ratio=0.1, maximum_ratio=0.95, reverse=False, details=False, lite=False, k=10,
        m=1):
    """
    FCV method
    :param original_model:
    :param X:
    :param y:
    :param minimum_ratio:
    :param maximum_ratio:
    :param reverse:
    :param shape:
    :param lite:
    :param k:
    :param m:
    :return:
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    if not reverse:
        arr1inds = y.argsort()
    else:
        arr1inds = y.argsort()[::-1]
    X = X[arr1inds]
    y = y[arr1inds]

    if not maximum_ratio:
        maximum_ratio = 1 - 1 / k
    sample_number = len(X)
    fold_sample_number = math.floor(sample_number / k)
    minimum_sample_number = round(minimum_ratio * sample_number)
    maxmum_sample_number = round(maximum_ratio * sample_number)

    label = []
    prediction = []
    max_value = []
    details = {
        "y_trains": [],
        "y_train_predicts": [],
        "y_tests": [],
        "y_test_predicts": [],
    }
    for split in range(fold_sample_number, sample_number, fold_sample_number if not lite else fold_sample_number * 10):
        if split < minimum_sample_number:
            continue
        if split > maxmum_sample_number:
            break

        #         print("Training 0 to {}, validation on {} to {}".format(split, split, split + fold_sample_number))
        sys.stdout.write("Training end in %s out of %s \r" % (split, sample_number))
        sys.stdout.flush()
        X_train = X[0:split]
        y_train = y[0:split]
        start_sample_number = split + (m - 1) * fold_sample_number
        end_sample_number = split + m * fold_sample_number

        if start_sample_number > sample_number:
            break
        else:
            end_sample_number = min(end_sample_number, sample_number)

        X_val = X[start_sample_number:end_sample_number]
        y_val = y[start_sample_number:end_sample_number]

        if issubclass(type(original_model), BaseEstimator):
            model = clone(original_model)
            model.fit(X_train, y_train)
        else:
            pass
            # if not hybrid:
            #     model = original_model(shape)
            #     model.fit(X_train, y_train, epochs=epochs, batch_size=128, verbose=0)
            # else:
            #     model = hybrid_train(original_model, X_train, y_train, shape)
        y_pred = model.predict(X_val)

        label.extend(list(y_val))
        prediction.extend(list(y_pred))
        max_value.extend([y[split - 1]] * len(y_val))
        if details:
            y_train_predict = model.predict(X_train)
            details['y_trains'].append(y_train)
            details['y_train_predicts'].append(y_train_predict)
            details['y_tests'].append(y_val)
            details['y_test_predicts'].append(y_pred)

        # if not issubclass(type(original_model), BaseEstimator):
        #     K.clear_session()
    if details:
        return label, prediction, details
    else:
        return label, prediction


def cv(original_model, X, y, shape=None, k=10):
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    if issubclass(type(original_model), BaseEstimator):
        cv_prediction = cross_val_predict(original_model, X, y, cv=KFold(k, shuffle=True))
    else:
        # fix random seed for reproducibility
        seed = 7
        np.random.seed(seed)

        # define 10-fold cross validation test harness
        kfold = KFold(n_splits=k, shuffle=True, random_state=seed)

        y_random = []
        cv_prediction = []
        for i, (train, test) in enumerate(kfold.split(X, y)):
            sys.stdout.write('{} of {} fold \r'.format(i + 1, k))
            sys.stdout.flush()

            # # train model
            if issubclass(type(original_model), BaseEstimator):
                model = clone(original_model)
                model.fit(X[train], y[train])
            else:
                pass
            # if not hybrid:
            #     model = original_model(shape)
            #     model.fit(X[train], y[train], epochs=epochs, batch_size=128, verbose=0)
            #     # validation_data=(X[test], y[test])
            # else:
            #     model = hybrid_train(original_model, X[train], y[train], shape)

            y_random.extend(y[test])
            cv_prediction.extend(model.predict(X[test]))
            # K.clear_session()

        y = y_random

    return y, cv_prediction


def forward_holdout_split(X, y, test_ratio, reverse=False):
    """
    :param X:
    :param y:
    :param test_ratio:
    :param reverse: False means y with high value as test set
    :return:
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    if not reverse:
        arr1inds = y.argsort()
    else:
        arr1inds = y.argsort()[::-1]
    X = X[arr1inds]
    y = y[arr1inds]
    sample_number = len(X)
    split = round((1 - test_ratio) * sample_number)
    X_train = X[0:split]
    y_train = y[0:split]
    X_test = X[split:]
    y_test = y[split:]
    return X_train, y_train, X_test, y_test


def forward_holdout(original_model, X, y, test_ratio, reverse=False):
    """
    holdout
    :param original_model:
    :param X:
    :param y:
    :param test_ratio:
    :param reverse:
    :return:
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    if not reverse:
        arr1inds = y.argsort()
    else:
        arr1inds = y.argsort()[::-1]
    X = X[arr1inds]
    y = y[arr1inds]
    sample_number = len(X)
    split = round((1 - test_ratio) * sample_number)
    X_train = X[0:split]
    y_train = y[0:split]
    X_val = X[split:]
    y_val = y[split:]
    if issubclass(type(original_model), BaseEstimator):
        model = clone(original_model)
        model.fit(X_train, y_train)
    else:
        pass
        # if not hybrid:
        #     model = original_model(shape)
        #     model.fit(X_train, y_train, epochs=epochs, batch_size=128, verbose=0)
        # else:
        #     model = hybrid_train(original_model, X_train, y_train, shape)
    y_pred = model.predict(X_val)
    return y_val, y_pred, X_train, y_train


def cal_metric(y_true, y_predict):
    """
    y_true:
    y_predict:
    return  metrics MSE RMSE MAE R2 R MRE
    """
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    n = len(y_true)
    MSE = mean_squared_error(y_true, y_predict)
    RMSE = pow(MSE, 0.5)
    MAE = mean_absolute_error(y_true, y_predict)
    R2 = r2_score(y_true, y_predict)
    pccs = pearsonr(y_true, y_predict)[0]
    # mre = mean_relative_error(y_true, y_predict)
    return dict({"n": n, "MSE": MSE, "RMSE": RMSE, "MSE": MSE, "MAE": MAE, "R2": R2, 'R': pccs,  # 'MRE': mre
                 })


def model_efficiency(y_extra_predict, y_inter_predict, task='high', extra_ratio=None, pecentage = 10):
    """
    :param y_extra_pred:
    :param y_inter_pred:
    :param task: high for higher task; low for lower task
    :param extra_ratio:
    :return:
    """

    if extra_ratio is not None:
        assert 0 < extra_ratio < 1

    assert task == 'high' or task == 'low'

    n_extra = len(y_extra_predict)
    n_inter = len(y_inter_predict)
    n_total = (n_extra + n_inter)
    if extra_ratio is None:
        extra_ratio = n_extra / n_total
    if task == 'high':
        c = np.percentile(y_extra_predict, pecentage)
        count_extra = len(np.where(y_extra_predict >= c)[0])
        count_inter = len(np.where(y_inter_predict >= c)[0])
        me = (count_extra / (count_extra + count_inter)) / extra_ratio
    else:
        c = np.percentile(y_extra_predict, 100-pecentage)
        count_extra = len(np.where(y_extra_predict <= c)[0])
        count_inter = len(np.where(y_inter_predict <= c)[0])
        me = (count_extra / (count_extra + count_inter)) / extra_ratio
    # print(c)
    # print(count_extra)
    # print(count_inter)
    return me
