# Import Required Libraries
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from Core.FeatureSelection import Selector


class Workflow(object):

    def __init__(self, input_path, file_format, target, features, missing_value, search_value, job_type,
                 feature_selection, feature_scaling, encode_categorical, binary_transform, categorical_features):
        self.__input_path = input_path
        self.__file_format = file_format
        self.__target = target
        self.__features_x = features
        self.__missing_value = missing_value
        self.__search_value = search_value
        self.__job_type = job_type
        self.__learning_object = None
        self.__feature_selection = feature_selection
        self.__feature_scaling = feature_scaling
        self.__encode_categorical = encode_categorical
        self.__binary_transform = binary_transform
        self.__categorical_features = categorical_features
        self.y_pred = list()

    def generate_score(self, job_type, y_test, y_pred):
        """
        Generated acore_matrix and return as dictionary
        :param job_type: type of model that you are trying to apply
            It could be any one of below
            - Regression
            - Classification
        :param y_test: target variable test
        :param y_pred: target variable predicted
        :return: score_matrix
        """
        score_dict = {}
        if job_type.lower() == "regression":
            score_dict["Normal_Accuracy"] = r2_score(y_test, y_pred)
            score_dict["Normal_MSE"] = mean_squared_error(y_test, y_pred)
            # score_dict["Intercept"] = self.__learning_object.intercept_
            # score_dict["Coefficients"] = self.__learning_object.coef_
        elif job_type.lower() == "classification":
            raise Exception("TODO: Development in Progress")
        else:
            raise TypeError('Given job_type "{}" not supported currently'.format(self.__job_type))
        return {"GENERAL_SUMMARY": score_dict}

    def run_estimator(self, model):
        """
        Runs given model and returns score_matrix
        :param model: input model that needs to be evaluated
        :return: score_matrix
        """
        # Read file path and store it into df variable
        df = model.read_dataframe(self.__input_path, self.__file_format)

        # Apply DDL on dataframe
        if len(self.__features_x) != 1 and 'ALL' not in self.__features_x:
            df = model.df_ddl(df=df, target=self.__target, column_ordering=True, features=self.__features_x)

        # Replace Missing values with Mean/Most Occurred according to the data type of the column
        """
        Below shows how to use replace_missing_values function 
        Note: In columns you can use either list of columns or ['ALL'] as a input value"""
        if self.__missing_value:
            if not isinstance(self.__search_value, str):
                raise TypeError("Entered Value must be a string value not a type of {}".format(
                    type(self.__search_value)))

            df = model.replace_missing_values(df, search_value=self.__search_value, replace_value=np.NaN,
                                              columns=self.__features_x)
            df = model.impute_dataframe(df)

        if self.__encode_categorical:
            # TODO: Need to work on returning the encode_dict and use it in model decoding, currently only
            #  capturing the details
            df, encode_dict = model.encode_categorical_features(df, self.__categorical_features,
                                                                self.__binary_transform)

        # Created Matrix of Features of X and target variable Y
        x, y = model.create_matrix_of_fetures(df, self.__features_x, self.__target)

        # Scale variable
        if self.__feature_scaling:
            x = model.scale_matrix_of_features(x)

        # Apply Feature Selection if possible
        if self.__feature_selection is not None:
            if not isinstance(self.__feature_selection, str):
                raise TypeError('feature_selection could be either BE or FS, not "{}"'.format(self.__feature_selection))

            if not self.__feature_scaling:
                raise Exception('Features must be scaled before applying feature_selection')

            if self.__feature_selection.upper() == 'BE':
                feature_obj = Selector(x, y)
                x = feature_obj.backward_elimination(df.columns)
            else:
                # TODO: Need to work on forward selection process
                raise TypeError('Given feature_selection "{}" not supported currently'.format(self.__feature_selection))

        if type(model).__name__ == "POLY":
            x = model.apply_polynomial_features(x, y)

        # Split train/test sets
        x_train, x_test, y_train, y_test = model.train_test_splitter(x, y)

        # Train model with given model param
        self.__learning_object = model.build_model(x_train, y_train)

        # Predict target values
        self.y_pred = self.__learning_object.predict(x_test)

        # Generate Score and accuracy
        score_matrix = self.generate_score(self.__job_type, y_test, self.y_pred)

        # Generate Cross validation score
        score_matrix.update(model.get_cross_validation_score(x_train, y_train))

        # Apply GridSearch on dataset to predict Optimized param values
        score_matrix.update(model.get_grid_search_summary(x_train, y_train, x_test, y_test))

        return score_matrix
