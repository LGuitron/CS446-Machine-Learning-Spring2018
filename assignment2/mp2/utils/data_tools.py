"""Implements feature extraction and data processing helpers.
"""


import numpy as np


def preprocess_data(dataset,
                    feature_columns=[
                        'Id', 'BldgType', 'OverallQual'
                        'GrLivArea', 'GarageArea'
                    ],
                    squared_features=False,
                    ):
    """Processes the dataset into vector representation.

    When converting the BldgType to a vector, use one-hot encoding, the order
    has been provided in the one_hot_bldg_type helper function. Otherwise,
    the values in the column can be directly used.

    If squared_features is true, then the feature values should be
    element-wise squared.

    Args:
        dataset(dict): Dataset extracted from io_tools.read_dataset
        feature_columns(list): List of feature names.
        squred_features(bool): Whether to square the features.

    Returns:
        processed_datas(list): List of numpy arrays x, y.
            x is a numpy array, of dimension (N,K), N is the number of example
            in the dataset, and K is the length of the feature vector.
            Note: BldgType when converted to one hot vector is of length 5.
            Each row of x contains an example.
            y is a numpy array, of dimension (N,1) containing the SalePrice.
    """
    columns_to_id = {'Id': 0, 'BldgType': 1, 'OverallQual': 2,
                     'GrLivArea': 3, 'GarageArea': 4, 'SalePrice': 5}

    x = []
    y = []

    for row, value in dataset.items():

        # Store current row's features temporaly
        xRow = []

        # Add only the feature values passed as parameter
        for i in range(len(feature_columns)):

            # In the case of the building type call the helper
            # function and introduce a boolean array of size 5
            if(feature_columns[i] == 'BldgType'):
                bldType = one_hot_bldg_type(value[columns_to_id['BldgType']])
                for i in range(len(bldType)):
                    xRow.append(int(bldType[i]))

            # For other data types add the raw data value
            else:
                xRow.append(int(value[columns_to_id[feature_columns[i]]]))

        x.append(xRow)
        y.append(int(value[columns_to_id['SalePrice']]))

    x = np.array(x)
    y = np.array(y)

    # Square feature value if necessary
    if(squared_features):
        x = x**(2)

    processed_dataset = [x, y]
    return processed_dataset


def one_hot_bldg_type(bldg_type):
    """Builds the one-hot encoding vector.

    Args:
        bldg_type(str): String indicating the building type.

    Returns:
        ret(list): A list representing the one-hot encoding vector.
            (e.g. for 1Fam building type, the returned list should be
            [1,0,0,0,0].
    """
    type_to_id = {'1Fam': 0,
                  '2FmCon': 1,
                  'Duplx': 2,
                  'TwnhsE': 3,
                  'TwnhsI': 4,
                  }

    ret = np.zeros(5)
    ret[type_to_id[bldg_type]] = 1
    return ret
