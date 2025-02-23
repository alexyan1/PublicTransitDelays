# Import required libraries
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV


class DimensionalityReduction(object):

    def __init__(self, model):
        self.__model = model

    def check_cross_validation(self, x_train, y_train, **kwargs):
        """
        Generic function that will apply cross validation on input model and generate Optimal score based on CV value
        :param x_train: input matrix of features x
        :param y_train: target variable y
        :param kwargs: input key-value arguments for cross_val_score
        :return: score_summary
        """
        score_summary = {}
        val_scores = cross_val_score(self.__model, x_train, y_train, **kwargs)
        #score_summary["cross-val-scores"] = val_scores
        score_summary["cross-accuracy"] = val_scores.mean()
        score_summary["cross-std-dev"] = val_scores.std()

        return score_summary

    def check_grid_search(self, x_train, y_train, x_test, y_test, params, **kwargs):
        """
        Generic function that will apply grid search on input model and results summary dictionary
        :param x_train: input matrix of features of X-train
        :param y_train: target variable y-train
        :param x_test: input matrix of features of X-test
        :param y_test: target variable y-test
        :param params: input param list for the model to apply and evaluate
        :param kwargs: input key-value arguments foe the GridSearchCV module
        :return: score_summary
        """
        if not isinstance(params, (dict, list)):
            raise TypeError("params must be a type of dictionary, not type of {}".format(type(params)))

        score_summary = {}
        grid = GridSearchCV(self.__model, params, **kwargs)
        grid.fit(x_train, y_train)
        score_summary["gs-best-accuracy"] = grid.best_score_
        score_summary["gs-best-rsme"] = np.mean((grid.predict(x_test) - y_test) ** 2)
        score_summary["gs-best-params"] = grid.best_params_

        return score_summary
