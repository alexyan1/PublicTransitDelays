# Constant Variables used in model

# Grid Search Params
MLR_PARAMS = {'fit_intercept': [True, False],  'normalize': [True, False],  'copy_X': [True, False]}
SVR_PARAMS = [{'C': [1, 10, 100, 1000],
               'kernel': ['linear']},
              {'C': [1, 5, 7, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
               'kernel': ['rbf'],
               'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
DT_PARAMS = [{'criterion': ['mse', 'mae'],
              'min_samples_split': [5, 10, 20, 30, 40],
              'max_depth': [2, 6, 8, 10, 12, 14, 16, 18, 20],
              'min_samples_leaf': [15, 17, 20, 40],
              'max_leaf_nodes': [10, 20, 30, 40]}]

RFR_PARAMS = [{'bootstrap': [True],
               'max_features': ['auto', 'sqrt', 'log2'],
               'criterion': ['mse', 'mae'],
               'min_samples_split': [5, 10, 20, 30, 40],
               'max_depth': [2, 6, 8, 10, 12, 14, 16, 18, 20],
               'min_samples_leaf': [15, 17, 20, 40],
               'max_leaf_nodes': [10, 20, 30, 40]}]

# TODO: Need to fine tune SRV params as list.
# SVR_PARAMS = {'C': [1, 5, 7, 10, 20, 30, 40, 50, 100, 1000], 'kernel': ['linear', 'rbf'],
# 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
