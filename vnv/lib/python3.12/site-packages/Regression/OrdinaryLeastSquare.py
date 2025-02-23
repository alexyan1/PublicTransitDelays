# Import required libraries
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import Core.Constants as cs
from Core.PreProcessing import Skeleton
from Core.FeatureExtraction import DimensionalityReduction


class MLR(Skeleton, DimensionalityReduction):

    # Constant values for the model
    params = cs.MLR_PARAMS

    def __init__(self):
        self.estimator = LinearRegression()
        Skeleton().__init__()
        DimensionalityReduction.__init__(self, self.estimator)

    def build_model(self, x_train, y_train, **kwargs):
        """
        Builds Linear Regression model with given kvargs
        :param x_train: input features of matrix X
        :param y_train: input dependent variable y
        :param kwargs: key-vale parameters passed to LinearRegression object
        :return: estimator Model
        """
        try:
            self.estimator = LinearRegression(**kwargs)
            self.estimator.fit(x_train, y_train)
            return self.estimator
        except Exception as exp:
            raise Exception('CLASS: MultiLinearRegression: {}'.format(exp))

    def get_cross_validation_score(self, x_train, y_train, **kwargs):
        """
        Function that will apply cross validation on input model and generate Optimal score based on CV value
        :param x_train: input matrix of features x
        :param y_train: target variable y
        :param kwargs: input key-value arguments for cross_val_score
        :return: score_summary
        """
        kv_summary = super().check_cross_validation(x_train, y_train, **kwargs)
        return {"CROSS_VALIDATION": kv_summary}

    def get_grid_search_summary(self, x_train, y_train, x_test, y_test, **kwargs):
        """
        Function that will apply grid search on input model and results summary dictionary
        :param x_train: input matrix of features of X-train
        :param y_train: target variable y-train
        :param x_test: input matrix of features of X-test
        :param y_test: target variable y-test
        :param params: input param list for the model to apply and evaluate
        :param kwargs: input key-value arguments foe the GridSearchCV module
        :return: score_summary
        """
        gs_summary = super().check_grid_search(x_train, y_train, x_test, y_test, self.params, **kwargs)
        return {"GRID_SEARCH": gs_summary}


class POLY(MLR):

    def __init__(self):
        MLR().__init__()
        self.estimator = LinearRegression()
        DimensionalityReduction.__init__(self, self.estimator)

    @staticmethod
    def apply_polynomial_features(x, y, degree=3, **kwargs):
        """
        Convert linear equation to polynomial features
        :param x: Input matrix X
        :param y: input vector y
        :param degree: degree of polymorphic features to be applied on x
        :param kwargs: list of key value arguments for PolynomialFeature function
        :return: matrix of features X with polynomial features
        """
        poly_feature = PolynomialFeatures(degree=degree, **kwargs)
        x = poly_feature.fit_transform(x)
        poly_feature.fit(x, y)
        return x
