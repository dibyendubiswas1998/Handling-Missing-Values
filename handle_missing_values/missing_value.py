import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from statistics import stdev

def drop_na(data_path):
    """
        delete all the rows, if all columns values are missing.
    """
    try:
        data = pd.read_csv(data_path, na_values=['NAN', 'NA', 'NO', 'No', 'NaN', 'NoT'])
        # get missing values info before handling
        before_ = data.isnull().sum().to_dict()
        data.dropna(how='all', inplace=True)
        # get missing values info after handling
        after_ = data.isnull().sum().to_dict()
        return before_, after_, data

    except Exception as ex:
        print(ex)
        raise ex   


def forward_backward_fill(data_path:str, cols:list, method:str):
    """
        this helps to replace the missing values using forward and backward fill method.
    """
    try:
        data = pd.read_csv(data_path, na_values=['NAN', 'NA', 'NO', 'No', 'NaN', 'NoT'])
        # get missing values info before handling
        before_ = data.isnull().sum().to_dict()
        # iterate one by one column
        for col in cols:
            data[col] = data[col].fillna(method=method)
        # get missing values info after handling
        after_ = data.isnull().sum().to_dict()
        return before_, after_, data
    
    except Exception as ex:
        print(ex)
        raise ex


def Mean_Median_Mode(data_path:str, cols:list, method:str):
    """
        this helps to replace the missing values using mean, median and mode.
    """
    try:
        data = pd.read_csv(data_path, na_values=['NAN', 'NA', 'NO', 'No', 'NaN', 'NoT'])
        print(data.dtypes)
        # get missing values info before handling
        before_ = data.isnull().sum().to_dict()
        for col in cols:
            # replace with mean value
            if method == 'mean':
                data[col] = data[col].fillna(data[col].mean())
            # replace with median value
            if method == 'median':
                data[col] = data[col].fillna(data[col].median())
            # replace with mode value
            if method == 'mode':
                data[col] = data[col].fillna(data[col].mode())
        # get missing values info after handling
        after_ = data.isnull().sum().to_dict()
        return before_, after_, data

    except Exception as ex:
        print(ex)
        raise ex


def Arbitary_Value(data_path:str, col:str, val):
    """
        this helps to replace the missing values using any arbitary value
    """
    try:
        data = pd.read_csv(data_path, na_values=['NAN', 'NA', 'NO', 'No', 'NaN', 'NoT'])
        # get missing values info before handling
        before_ = data.isnull().sum().to_dict()
        # replace with any arbitary value
        data[col] = data[col].fillna(val)
        # get missing values info after handling
        after_ = data.isnull().sum().to_dict()
        return before_, after_, data

    except Exception as ex:
        print(ex)
        raise ex


def Random_Sample_imputation(data_path:str, cols:str):
    """
        this helps to handle the missing values with random sample imputation 
    """
    data = pd.read_csv(data_path, na_values=['NAN', 'NA', 'NO', 'No', 'NaN', 'NoT'])
    # get missing values info before handling
    before_ = data.isnull().sum().to_dict()
    for col in cols:
        random_sample = data[col].dropna().sample(data[col].isnull().sum(),random_state=0)
        random_sample.index = data[data[col].isnull()].index
        data.loc[data[col].isnull(), col] = random_sample
    # get missing values info after handling
    after_ = data.isnull().sum().to_dict()
    return before_, after_, data


def Removing_rows_cols(data_path:str, rows_num:list=None, cols:list=None):
    """
        this helps to handle the missing values, through removing rows or columns.
    """
    try:
        data = pd.read_csv(data_path, na_values=['NAN', 'NA', 'NO', 'No', 'NaN', 'NoT'])
        # get missing values info before handling
        before_ = data.isnull().sum().to_dict()
        # deleting the rows
        if rows_num:
            data.drop(rows_num, axis=0, inplace=True)
        # deleting the columns
        if cols:
            data.drop(cols, axis=1, inplace=True)
        # get missing values info after handling
        after_ = data.isnull().sum().to_dict()
        return before_, after_, data
            
    except Exception as ex:
        print(ex)
        raise ex


def Apply_KNN_Imputer(data_path:str, cols:list, n_neighbors:int=2):
    """
        this helps to handle the missing values using KNN Imputer.
    """
    try:
        data = pd.read_csv(data_path, na_values=['NAN', 'NA', 'NO', 'No', 'NaN', 'NoT'])
        # get missing values info before handling
        before_ = data.isnull().sum().to_dict()
        # apply KNN Imputation
        imputer = KNNImputer(n_neighbors=n_neighbors)
        data[cols] = imputer.fit_transform(data[cols])
        # get missing values info after handling
        after_ = data.isnull().sum().to_dict()
        return before_, after_, data

    except Exception as ex:
        print(ex)
        raise ex


def Third_std(data_path:str, cols:list):
    """
        this helps to handle the missing value, using above 3rd std data points.
    """
    try:
        data = pd.read_csv(data_path, na_values=['NAN', 'NA', 'NO', 'No', 'NaN', 'NoT'])
        # get missing values info before handling
        before_ = data.isnull().sum().to_dict()
        for col in cols:
            std = data[col].describe().loc['std']
            mean = data[col].describe().loc['mean']
            above_3rd_std = mean + (3 * std) + 1
            below_3rd_std = mean - (3 * std) + 1
            # fill the na values with above 3rd std
            data[col] = data[col].fillna(above_3rd_std)

        # get missing values info after handling
        after_ = data.isnull().sum().to_dict()
        return before_, after_, data

    except Exception as ex:
        print(ex)
        raise ex



if __name__ == "__main__":
    pass
