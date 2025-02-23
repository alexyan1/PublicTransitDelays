# Import required libraries
from abc import abstractmethod
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder


class Skeleton(object):

    def __init__(self):
        self.__fill = None

    @staticmethod
    def read_dataframe(input_path, file_format='csv', **kwargs):
        """
        Reads input file and return pandas dataframe
        :param input_path: path where we the file is located
        :param file_format: extension of the file to be read
        :param kwargs: list of key-value arguments to be passed to pd.read_() function
        :return: pandas dataframe
        """
        df = pd.DataFrame()
        try:
            if file_format == 'csv':
                df = pd.read_csv(input_path, **kwargs)
        except FileNotFoundError:
            raise FileNotFoundError('File Not found at specified @ {}'.format(input_path))
        except Exception as exp:
            raise Exception('FUNCTION: read_dataframe: {}'.format(exp))
        return df

    @staticmethod
    def df_ddl(df, target, column_ordering=False, features=[], drop_columns=[]):
        """
        Alters pandas dataframe like Column ordering, dropping, rename and etc...
        :param df: input pandas dataframe
        :param target: Target variable
        :param drop_columns: list of columns to be dropped
        :param column_ordering: If true ordering columns will be done
        :param features: ordered column list, will be applicable when 'column_ordering' is set to True
        :return: altered pandas dataframe
        """
        # Raise Exception if df is not a type of PANDAS dataframe
        if not isinstance(df, pd.DataFrame):
            raise TypeError('df: must be a pandas dataframe, not a type of {}'.format(type(df)))

        # Drops column if we have any value in drop_columns param
        if not isinstance(drop_columns, list):
            raise TypeError('{} must be a instance of list or None type, not a type of {}'.format('drop_columns',
                                                                                                  type(drop_columns)))
        else:
            for col in drop_columns:
                try:
                    df = df.drop(col, axis=1)
                except KeyError:
                    raise KeyError('Key "{}" not found in list dataframe columns\n{}'.format(col, df.columns))

        # Orders column in given order if colum_ordering is TRUE:
        if column_ordering:
            if not isinstance(features, list):
                raise TypeError('{} must be a list not a type of {}'.format('column_list', type(features)))
            else:
                try:
                    if len(features) != 1:
                        df = df[features + [target]]
                    else:
                        df = df[features]
                except Exception as exp:
                    raise Exception('FUNCTION: df_ddl: {}'.format(exp))
        return df

    def create_matrix_of_fetures(self, df, X_columns, y_column):
        """
        Creates Matrix of features of X and target variable y from given dataframe
        :param df: input dataframe which needs to be converted as X and Y
        :param X_columns: list of columns for X
        :param y_column: y column name
        :return: matrix of feature
        """
        if len(X_columns) != 1 and 'ALL' not in X_columns:
            x = df[X_columns].iloc[:, :].values.astype(np.float)
        else:
            X_columns = [col for col in df.columns if col != y_column]
            x = df[X_columns].iloc[:, :].values.astype(np.float)
        col_pos = df.columns.get_loc(y_column)
        y = np.asarray(df.iloc[:, col_pos]).astype(np.float)
        return x, y

    @staticmethod
    def train_test_splitter(x, y, **kwargs):
        """
        Splits given matrix of features in to Train/Test sets
        :param x: number of columns[column names] in Independent variable
        :param y: dependent variable name
        :param kwargs: input key-value arguments for train_test_split function
        :return: splitter train/test dataset of numpy array/matrix
        """
        try:
            x_train, x_test, y_train, y_test = train_test_split(x, y, **kwargs)
        except Exception as exp:
            raise Exception('FUNCTION: train_test_splitter: {}'.format(exp))
        return x_train, x_test, y_train, y_test

    @staticmethod
    def replace_missing_values(df, search_value, replace_value=np.NaN, columns=[]):
        """
        Replace Unknown values in dataframe
        :param df: input dataframe where missing data handling to be applied
        :param search_value: missing value to be search and replaced
        :param replace_value: replace value to be replaced with search value
        :param columns: columns list where we need to search for the missing value
        :return: pandas dataframe with replaced values
        """
        # Raise Exception if df is not a type of PANDAS dataframe
        if not isinstance(df, pd.DataFrame):
            raise TypeError('df: must be a pandas dataframe, not a type of {}'.format(type(df)))

        # Raise Exception if columns is not a type of list
        if not isinstance(columns, list):
            raise TypeError('columns: must be a list, not a type of {}'.format(type(columns)))

        if 'ALL' in columns and len(columns) == 1:
            columns = df.columns.values

        for col in columns:
            if not df[df[col].isin([search_value])].empty:
                print("Found search string '{}' in column '{}', replacing to '{}'".format(search_value, col,
                                                                                          replace_value))
                if isinstance(replace_value, type(np.NaN)):
                    df[col] = df[col].replace(search_value, np.NaN)
                else:
                    df[col] = df[col].replace(search_value, replace_value)

        return df

    def impute_dataframe(self, df):
        """
        Impute pandas dataframe to hanle missing values
        :param df: input pandas dataframe to be imputed[Handle missing value]
        :return: imputed Pandas dataframe
        """
        self.__fill = pd.Series({df[c].name: df[c].value_counts().index[0] if df[c].dtype == np.dtype('O') else df[c].mean()
                               for c in df})

        return df.fillna(self.__fill)

    @staticmethod
    def encode_categorical_features(df, features, binary_transform=False):
        """
        Encodes Categorical data in to integer/Binary format
        :param df: input dataframe with categorical features to be encoded
        :param features: list of categorical columns in dataframe
        :param binary_transform: If true will encode the categorical data to dummy variables
        :return: encoded dataframe with categorical features list
        """
        encode_dict = {}
        if not isinstance(features, list):
            raise TypeError('categorical_features must be a list, not a type of {}'.format(type(features)))

        for col in features:
            new_set = set()
            df["{}_encoded".format(col)] = LabelEncoder().fit_transform(df[col])
            df[[col, "{}_encoded".format(col)]].apply(
                lambda row: new_set.add((row[col], row["{}_encoded".format(col)])), axis=1)
            encode_dict[col] = new_set
            if binary_transform:
                df[col] = pd.Categorical(df[col])
                encoded_df = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = encoded_df.join(df)
                df = df.drop("{}_encoded".format(col), axis=1)
                df = df.drop(col, axis=1)
            else:
                df[col] = df["{}_encoded".format(col)]
                df = df.drop("{}_encoded".format(col), axis=1)

        return df, encode_dict

    @abstractmethod
    def build_model(self):
        """
        Abstract method used to build model building logic in different child models
        :return: Trained Machine Learning Model from sklearn library
        """
        pass

    @staticmethod
    def scale_matrix_of_features(x, **kwargs):
        """
        apply scaling in given matrix of features of x
        :param x: matrix of feautes of x
        :param kwargs: key value arguments for StandardScaler function
        :return: scaled matrix of features of x
        """
        x_scalar = StandardScaler(**kwargs)
        x = x_scalar.fit_transform(x)
        return x
