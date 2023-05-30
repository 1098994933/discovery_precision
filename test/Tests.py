import unittest
import sklearn.metrics
from util.eval.eval_method import fcv, cv, forward_holdout, forward_holdout_split
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from util.ml.metrics import model_efficiency
from sklearn.metrics import mean_absolute_error, r2_score


class EvalMethodTest(unittest.TestCase):

    @staticmethod
    def test_fcv():
        """test the Evaluation Method"""
        X = np.array(list(range(100))).reshape(-1, 1)
        y = np.array(list(range(100)))
        original_model = RandomForestRegressor()
        y_true, y_predict = fcv(original_model, X, y)
        print(y_true, y_predict)

    @staticmethod
    def test_fcv_details():
        """test the Evaluation Method"""
        X = np.array(list(range(100))).reshape(-1, 1)
        y = np.array(list(range(100)))
        original_model = RandomForestRegressor()
        y_true, y_predict, details = fcv(original_model, X, y, details=True)
        print(details)

    @staticmethod
    def test_get_score_by_details():
        X = np.array(list(range(100))).reshape(-1, 1)
        y = np.array(list(range(100)))
        original_model = RandomForestRegressor()
        y_true, y_predict, details = fcv(original_model, X, y, k=10, details=True)
        for i in range(len(details['y_trains'])):
            print(model_efficiency(details['y_test_predicts'][i], details['y_train_predicts'][i], extra_ratio=0.1))
        metrics = [model_efficiency(details['y_test_predicts'][i], details['y_train_predicts'][i], extra_ratio=0.1) for
                   i in range(len(details['y_trains']))]
        metrics2 = [r2_score(details['y_test_predicts'][i], details['y_tests'][i]) for i in
                    range(len(details['y_trains']))]
        print(np.median(metrics))
        print(np.median(metrics2))

    @staticmethod
    def test_cv():
        X = np.array(list(range(100))).reshape(-1, 1)
        y = np.array(list(range(100)))
        original_model = RandomForestRegressor()
        y_true, y_predict = cv(original_model, X, y)
        print(y_true, y_predict)

    @staticmethod
    def test_holdout():
        X = np.array(list(range(100))).reshape(-1, 1)
        y = np.array(list(range(100)))
        original_model = RandomForestRegressor()
        y_true, y_predict, _, _ = forward_holdout(original_model, X, y, test_ratio=0.2)
        print(y_true, y_predict)

    @staticmethod
    def test_holdout_split():
        X = np.array(list(range(100))).reshape(-1, 1)
        y = np.array(list(range(100)))
        test_ratio = 0.2
        X_train, y_train, X_test, y_test = forward_holdout_split(X, y, test_ratio, reverse=False)
        print(len(X_train))
        print(max(y_train))

    @staticmethod
    def test_model_efficiency():
        me = model_efficiency([1, 2, 3, 4, 5, 6], [-1, -2, -3, -4, -5, -6])

        assert me == 2
        me = model_efficiency([1, 2, 3, 4, 5, 6], [-1, -2, -3, -4, -5, -6], task='low')
        assert me == 0

        me = model_efficiency([1, 2, 3, -9, -9, -9], [-1, -2, -3, -4, -5, -6], task='high')
        assert me == 1

        me = model_efficiency([1, 2, 3, -9, -9, -9], [9, 9, 9, -4, -5, -6], task='high')
        assert me == 1

        me = model_efficiency([1, 2, 3, -9, -9, -9], [9, 9, 9, -4, -5, -6, -1, -1], task='high')
        assert me == (3 / 6) / (6 / 14)


if __name__ == '__main__':
    unittest.main()
