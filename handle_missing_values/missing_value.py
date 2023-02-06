import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def drop_na(data_path):
    """
        delete all the rows, if all columns values are missing.
    """
    try:
        data = pd.read_csv(data_path)
        data.dropna(how='all', inplace=True)
        return data
    except Exception as ex:
        print(ex)
        raise ex   





if __name__ == "__main__":
    
    data = drop_na('Data/missing.csv')
    print(data)
    print(data.isnull().sum().to_dict())
    