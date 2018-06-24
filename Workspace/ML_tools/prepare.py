# This module is made to aid data analysis by providing commonly used functions
# for inspecting and analysing a dataset.

# Setup data processing dependencies
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Imputer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, ShuffleSplit
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, roc_curve, precision_recall_curve
from skimage.io import imread, imshow
from skimage.transform import rescale, resize, downscale_local_mean

# ===============================
# === Inspection and analysis ===

# Get dataframe containing information about the unique labels within the dataset column/array.
def datasets_overview(df_list, df_names):
    """
    Helper function that outputs a table containing high level/general statistics of the provided dataframes.

    Arguments:
    -----------
    df_list : pd.dataframe, list of dataframes as input
    df_names : list, list containing the names assigned to the dataframes in their respective order

    Returns:
    -----------
    df_overview : pd.dataframe, resulting dataframe as output
    """
    num_rows = []
    num_cols = []
    total_col_names = []

    # Iterate through dataframes
    for df in df_list:
        col_names = ''
        first_col = True
        # Iterate through column names
        for col_name in df.columns.tolist():
            if first_col == True:
                first_col = False
                col_names = col_name
            else:
                col_names = col_names + ', ' + col_name

        num_rows.append(df.shape[0])
        num_cols.append(df.shape[1])
        total_col_names.append(col_names)

    # Setup overview info
    df_overview = pd.DataFrame(
        data={
            '': dataset_names,
            'Number of entries': num_rows,
            'Number of features': num_cols,
            'Some feature names': total_col_names
        }
    )

    df_overview.set_index('', inplace=True)

    return df_overview

# Compare and find common column features between given dataframes.
def compare_common(df_list):
    """
    Helper function that compares any number of given dataframes and output their common column features.

    Arguments:
    -----------
    df_list : pd.dataframe, list of dataframes as input

    Returns:
    -----------
    df_common : pd.dataframe, dataframe detailing the common column features
    """
    common_col_names = []
    for df in df_list:
        if len(common_col_names) == 0:
            common_col_names = df.columns.tolist()
        else:
            common_col_names = list(set(common_col_names) & set(df.columns.tolist()))

    df_common = pd.DataFrame({'Features(s) common between the datasets': common_col_names})
    print('Number of features common between the given datasets: ' + str(len(common_col_names)))

    return df_common

# Compare and find different column features between a main and other given dataframes.
def compare_difference(df_other, df_train):
    """
    Helper function that compares a main dataframe (typically the training dataset) with a different dataframe and output their different column features with respect to the other dataframe.

    Arguments:
    -----------
    df_list : pd.dataframe, list of dataframes to be passed in as input

    Returns:
    -----------
    df_difference : pd.dataframe, dataframe detailing the different column features
    """
    train = df_train.columns.tolist()
    other = df_other.columns.tolist()
    col_names_union = list(set().union(other, train))
    col_names_intersect = list(set(other) & set(train))
    diff_col_names = list(set(col_names_union) - set(col_names_intersect) - set(df_train.columns.tolist()))

    df_difference = pd.DataFrame({'Features(s) not found in the training dataset': diff_col_names})
    print('Number of features not found the main dataset: ' + str(len(diff_col_names)))

    return df_difference

# Checks unique value labels, counts and the relative representations in the dataset.
def unique_values(df, header):
    """
    Helper function that outputs a table containing the unique label, the coresponding entries count and the relatively representation of the data in the dataset

    Arguments:
    -----------
    df : pd.dataframe, dataframe to be passed in as input
    header : string, the column header description to apply analysis on

    Returns:
    -----------
    df_unique : pd.dataframe, resulting dataframe as output
    """
    header_no = '(' + 'Unique values: ' + str(df[header].nunique()) + ')'
  
    # Create unique value result dataframe.
    df_unique = pd.DataFrame(
        {
            str(header_no) + ' ' + header  : df[header].value_counts().index.values,
            'Number of examples' : np.asarray(df[header].value_counts())
        }
    )
  
    # Relative prevalence of the data representation amongst all other categories.
    # (expressed in percentage proportion)
    label_count_sum = df_unique['Number of examples'].sum()
    df_unique['Relative representation %'] = 100*df_unique['Number of examples']/label_count_sum
  
    return df_unique

# Checks the unique value labels, counts and relative representations for the provided column headers in the dataset.
def relative_representation(df, header_list, min_threshold=100):
    """
    Helper function that outputs a summary table of categorical values where their relative prevelance is less than a given threshold (min_threshold)

    Arguments:
    -----------
    df : pd.dataframe, dataframe to be passed as input
    header_list : list, list of columns headers to be passed in as input
    min_threshold : float, the minimum threshold where the categorical value prevalence is considered to be adequately represented

    Returns:
    -----------
    df_relative_rep : pd.dataframe, resulting dataframe as output
    """
    # Create dataframe structure
    df_relative_rep = pd.DataFrame({'Column value': [], 'Number of entries': [
    ], 'Relative representation %': [], 'Column header': []})

    # Get all the relative representations of the given list of headers
    for header in header_list:
        df_unique = unique_values(df, header)
        df_unique = df_unique[df_unique['Relative representation %']
                              < min_threshold]
        df_unique['Column header'] = header
        df_unique.columns = ['Column value', 'Number of entries',
                             'Relative representation %', 'Column header']
        df_relative_rep = pd.concat([df_relative_rep, df_unique], axis=0)

    # Tidy up the output dataframe
    df_relative_rep.sort_values(
        by=['Relative representation %'], ascending=False, inplace=True)
    df_relative_rep.reset_index(inplace=True)
    df_relative_rep.drop(columns=['index'], inplace=True)

    return df_relative_rep

# Check the relative proportion of data that contain missing value in the dataset.
def missing_values_table(df):
    """
    Helper function that outputs a summary table of the proportion of data that contain missing value in the dataset

    Arguments:
    -----------
    df : pd.dataframe, dataframe to be passed as input

    Returns:
    -----------
    mis_val_table_ren_columns : pd.dataframe, resulting dataframe as output
    """
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
          "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns

# Secondary helper function for creating custom (binary) target labels.
def target_label(row, target_header, lookup_value, mode='equal to'):
    value = row[target_header]
    output = 0
    if 'greater than' in mode and (isinstance(value, float) or isinstance(value, int)):
        if value > lookup_value:
            output = 1

    if 'less than' in mode and (isinstance(value, float) or isinstance(value, int)):
        if value < lookup_value:
            output = 1

    if 'equal to' in mode:
        if value == lookup_value:
            output = 1

    if 'contains' in mode and isintance(value, str):
        if lookup_value in value:
            output = 1
    
    row[target_header + '_' + str(lookup_value)] = output
    
    return row

# Helper function to creating new target labels
def create_target(df, target_header, lookup_value, mode):
    """
    Helper function that outputs a table containing the newly created target label(s).

    Arguments:
    -----------
    df : pd.dataframe, dataframe to be passed as input
    target_header : string, the column with the header description to run feature correlations against
    lookup_value : the target value to be sought after in the existing target column
    mode : string, selection of 'equal to', 'greater than', 'less than', 'contains', the type of operator setting

    Returns:
    -----------
    df_output : pd.dataframe, resulting dataframe as output
    """
    df_output = df.apply(lambda row : target_label(row, target_header=target_header, lookup_value=lookup_value, mode=mode), axis=1)
    
    return df_output

# Feature analysis with correlations
def correlations_check(df, target_header, target_label=None, encoder=None):
    """
    Helper function that outputs a table of feature correlations against a specified column in the dataset

    Arguments:
    -----------
    df : pd.dataframe, dataframe to be passed as input
    target_header : string, the column with the header description to run feature correlations against
    target_label : string, optional input if the target column is not yet encoded with binary 0 and 1 labels
    encoder : selection of 'one_hot', 'label_encoding', the type of encoding method for categorical data

    Returns:
    -----------
    df_correlations : pd.dataframe, resulting dataframe as output
    """

    print('Inspecting data values... ', end='')
    # Get target data
    df_y = df[[target_header]]

    # Isolate sub-dataset containing categorical values
    categorical = df.loc[:, df.dtypes == object]
    
    # Isolate sub-dataset containing non-categorical values
    non_categorical = df.loc[:, df.dtypes != object]
    print('[Done]')

    # Apply encoding to categorical value data
    if encoder == 'one_hot':
        print('Encoding categorical data... ', end='')
        categorical = pd.get_dummies(categorical)
        print('[Done]')

    # Join up categorical and non-categorical sub-datasets
    df_x = pd.concat([categorical, non_categorical], axis=1)
        
    # Get the encoded target labels if necessary
    # Check if target labels are binary 0 and 1
    print('Inspect target data type... ', end='')
    binary_col_headers = get_binary_headers(df_y, [target_header])
    if target_header in binary_col_headers:
        y = df_y[target_header]
    else:
        # Else if column values not binary 0 and 1, proceed to encode target labels with one-hot encoding
        df_y = pd.get_dummies(df_y)

        # Select the relevant column of the specified target value as per input
        target_headers = df_y.columns.tolist()

        if target_label != None:
            for header in target_headers:
                if target_label in header:
                    y = df_y[header]
                    break
                else:
                    pass
        else:
            y = df_y.iloc[:, 0]
            print('Note: Target column contains multiple labels. \nThe column is one-hot encoded and the first column of the encoded result is selected as the target label for feature influence analysis.\n')
    print('[Done]')

    print('Extracting data correlations... ', end='')
    # Get the correlation values with respect to the target column
    df_correlations = pd.DataFrame(df_x.corr()[target_header].sort_values(ascending=False))
    print('[Done]')

    # Drop the row with the index of the target header (correlation value to this row is 1 - with itself)
    df_correlations.drop(df_correlations.index[df_correlations.index.get_loc(target_header)], inplace=True)

    # Drop rows containing NaN
    df_correlations.dropna(inplace=True)
    
    return df_correlations

# Feature analysis with PCA
def pca_check(df, target_header, encoder='one_hot', numerical_imputer=None, scaler=None, pca_components=10):
    """
    Helper function that outputs PCA transformation and the associated features contributions. Also outputs Scree plots on Eigenvalues and Explained variance attributes.

    Arguments:
    -----------
    df : pd.dataframe, dataframe to be passed as input
    target_header : string, the column with the header description of the target label
    encoder : selection of 'one_hot', 'label_encoding', the type of encoding method for categorical data
    numerical_imputer : selection of 'mean', 'median', 'most_frequent', the type of imputation strategy for processing missing data
    scaler : string, selection of 'standard', 'minmax' or 'robust', type of scaler used for data processing
    pca_components : int, the number of principal components to be extracted from PCA

    Returns:
    -----------
    df_pca : pd.dataframe, resulting dataframe of PCA transformed data as output
    df_pca_comp : pd.dataframe, resulting dataframe of PCA associated features contributions as output
    Display of Scree plots (Eigenvalue and Explained variance)
    """

    print('Inspecting data values... ', end='')
    # Separate features and target data
    df_y = df[[target_header]].copy()
    df_x = df.drop(columns=[target_header]).copy()
    
    # Isolate sub-dataset containing categorical values
    categorical = df_x.loc[:, df_x.dtypes == object].copy()
    
    # Isolate sub-dataset containing non-categorical values
    non_categorical = df_x.loc[:, df_x.dtypes != object].copy()
    binary_col_headers = get_binary_headers(non_categorical, non_categorical.columns.tolist())
    non_categorical.drop(columns=binary_col_headers, inplace=True)
    non_categorical_headers = non_categorical.columns.tolist()
    print('[Done]')

    # Apply numerical imputation processing to data
    if numerical_imputer != None:
        print('Imputing numerical data... ', end='')
        numerical_imputation = Imputer(strategy=numerical_imputer)
        non_categorical = numerical_imputation.fit_transform(non_categorical)
        print('[Done]')

    # Apply scaler to data
    if scaler != None:
        print('Scaling numerical data... ', end='')
        if scaler == 'standard':
            scaler = StandardScaler()
            scaler.fit(non_categorical)
        elif scaler == 'minmax':
            scaler = MinMaxScaler()
            scaler.fit(non_categorical)
        else:
            scaler = RobustScaler()
            scaler.fit(non_categorical)
        non_categorical_data = scaler.transform(non_categorical)
        non_categorical = pd.DataFrame(data=non_categorical_data, columns=non_categorical_headers)
        print('[Done]')

    # Apply encoding to categorical value data
    if encoder != None:
        print('Encoding categorical data... ', end='')
        if encoder == 'one_hot':
            categorical = pd.get_dummies(categorical)
        print('[Done]')

    # Join up categorical and non-categorical sub-datasets
    df_x = pd.concat([categorical, non_categorical], axis=1)
    feature_headers = df_x.columns

    print('Applying PCA transformation... ', end='')
    # Fit data to PCA transformation of specified principal components
    pca = PCA(n_components=int(pca_components))
    pca.fit(df_x)
    x_pca = pca.transform(df_x)

    # Set header descriptions for displaying PCA results
    pc_headers = ['PC_' + str(pc_index + 1) for pc_index in range(pca_components)]
    df_x_pca = pd.DataFrame(data=x_pca, columns=pc_headers)
    
    # Join up categorical and non-categorical sub-datasets
    df_pca = pd.concat([df_x_pca, df_y], axis=1)
    
    # Set table containing PCA components contribution
    df_pca_comp = pd.DataFrame(data=pca.components_, columns=feature_headers)
    print('[Done]')

    # Get PCA eigenvalues, explained variance ratio, and cumulative explained variance of top components
    df_explained_var = pd.DataFrame(data=pca.explained_variance_ratio_, index=pc_headers, columns=['Explained variance %'])
    df_explained_var['Explained variance %'] = df_explained_var['Explained variance %']*100
    df_explained_var['Explained variance (cumulative) %'] = df_explained_var['Explained variance %'].cumsum()
    df_explained_var['Eigenvalue'] = pca.explained_variance_
    print('\nTotal explained variance %: {:0.2f}%\n'.format(df_explained_var['Explained variance %'].sum()))
    print(df_explained_var)
    print('\n')
    
    # Scree plots
    custom_rc = {'lines.linewidth': 0.8, 'lines.markersize': 0.8} 
    sns.set_style('white')
    sns.set_context('talk', rc=custom_rc)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    df_explained_var['PC']= [pc_index + 1 for pc_index in range(pca_components)]
    
    # Eigenvalue plot
    sns.pointplot(x='PC', y='Eigenvalue', data=df_explained_var, ax=ax1, color='Blue')
    ax1.set_title('Scree plot')
    ax1.set_xlabel('Principal component')
    ax1.set_ylabel('Eigenvalue')
    
    # Explained variance plot
    sns.pointplot(x='PC', y='Explained variance %', data=df_explained_var, ax=ax2, color='Blue')
    sns.pointplot(x='PC', y='Explained variance (cumulative) %', data=df_explained_var, ax=ax2, color='Grey')
    ax2.set_title('Explained variance')
    ax2.set_xlabel('Principal component')
    ax2.set_ylabel('Explained variance %')
    ax2.legend(labels=('Variance', 'Variance cumulative'))
    leg = ax2.get_legend()
    leg.legendHandles[0].set_color('Blue')
    leg.legendHandles[1].set_color('Grey')
    
    return df_pca, df_pca_comp

# Feature analysis with logistic regression
def svm_anomaly_features(df, target_header, target_label=None, encoder=None, numerical_imputer=None, scaler=None, nu=0.2):
    """
    Helper function that outputs feature weights from the trained one-class SVM model (with linear kernel).

    Arguments:
    -----------
    df : pd.dataframe, dataframe to be passed as input
    target_header : string, the column with the header description of the target label
    target_label : string, optional input if the target column is not yet encoded with binary 0 and 1 labels
    encoder : selection of 'one_hot', 'label_encoding', the type of encoding method for categorical data
    numerical_imputer : selection of 'mean', 'median', 'most_frequent', the type of imputation strategy for processing missing data
    scaler : string, selection of 'standard', 'minmax' or 'robust', type of scaler used for data processing
    nu : float, regularization parameter for one-class SVM where it is upper bounded at the fraction of outliers and a lower bounded at the fraction of support vectors

    Returns:
    -----------
    df_features : pd.dataframe, resulting dataframe of model feature weights as output
    """

    print('Inspecting data values... ', end='')
    # Separate features and target data
    df_y = df[[target_header]].copy()
    df_x = df.drop(columns=[target_header]).copy()
    
    # Isolate sub-dataset containing categorical values
    categorical = df_x.loc[:, df_x.dtypes == object].copy()
    
    # Isolate sub-dataset containing non-categorical values
    non_categorical = df_x.loc[:, df_x.dtypes != object].copy()
    binary_col_headers = get_binary_headers(non_categorical, non_categorical.columns.tolist())
    non_categorical.drop(columns=binary_col_headers, inplace=True)
    non_categorical_headers = non_categorical.columns.tolist()
    print('[Done]')

    # Apply numerical imputation processing to data
    if numerical_imputer != None:
        print('Imputing numerical data... ', end='')
        numerical_imputation = Imputer(strategy=numerical_imputer)
        non_categorical = numerical_imputation.fit_transform(non_categorical)
        print('[Done]')

    # Apply scaler to data
    if scaler != None:
        print('Scaling numerical data... ', end='')
        if scaler == 'standard':
            scaler = StandardScaler()
            scaler.fit(non_categorical)
        elif scaler == 'minmax':
            scaler = MinMaxScaler()
            scaler.fit(non_categorical)
        else:
            scaler = RobustScaler()
            scaler.fit(non_categorical)
        non_categorical_data = scaler.transform(non_categorical)
        non_categorical = pd.DataFrame(data=non_categorical_data, columns=non_categorical_headers)
        print('[Done]')

    # Apply encoding to categorical value data
    if encoder != None:
        print('Encoding categorical data... ', end='')
        if encoder == 'one_hot':
            categorical = pd.get_dummies(categorical)
        print('[Done]')
        
    # Join up categorical and non-categorical sub-datasets
    df_x = pd.concat([categorical, non_categorical], axis=1)
    feature_headers = df_x.columns
    X = df_x.values
    
    # Get the encoded target labels if necessary
    # Check if target labels are binary 0 and 1
    print('Inspect target data type... ', end='')
    binary_col_headers = get_binary_headers(df_y, [target_header])
    if target_header in binary_col_headers:
        y = df_y[target_header]
    else:
        # Else if column values not binary 0 and 1, proceed to encode target labels with one-hot encoding
        df_y = pd.get_dummies(df_y)

        # Select the relevant column of the specified target value as per input
        target_headers = df_y.columns.tolist()

        if target_label != None:
            for header in target_headers:
                if target_label in header:
                    y = df_y[header]
                    break
                else:
                    pass
        else:
            y = df_y.iloc[:, 0]
            print('Note: Target column contains multiple labels. \nThe column is one-hot encoded and the first column of the encoded result is selected as the target label for feature influence analysis.\n')
    print('[Done]')

    # Split train and test data for model fitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    print('Training model... no. of training examples: ' + str(X_train.shape[0]) + ', no. of features: ' + str(X_train.shape[1]) + '. ', end='')
    # Perform model training and evaluation
    model = OneClassSVM(nu=nu, kernel='linear')
    model.fit(X_train, y_train)
    print('[Done]')
    
    # Get the model performance
    y_pred = model.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    df_positive_rate = pd.DataFrame({'False positive rate' : fpr, 'True positive rate' : tpr})

    # Get the important features from the model in a dataframe format
    df_features = pd.DataFrame(data=model.coef_, columns=feature_headers).transpose()
    df_features.columns=['Feature weight']
    df_features.sort_values(by=['Feature weight'], ascending=False, inplace=True)
    
    print('\nOne-class SVM model with linear kernel evaluation:\n')
    print(classification_report(y_test, y_pred))

    # ROC plot
    plt.figure(figsize=(8, 6))
    custom_rc = {'lines.linewidth': 0.8, 'lines.markersize': 0.8} 
    sns.set_style('white')
    sns.set_context('talk', rc=custom_rc)
    ax = sns.pointplot(x='False positive rate', y='True positive rate', data=df_positive_rate, color='Blue')
    ax.set_title('ROC plot')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')

    return df_features

# Feature analysis with logistic regression
def logistic_reg_features(df, target_header, target_label=None, encoder=None, numerical_imputer=None, scaler=None, reg_C=10, reg_norm='l2'):
    """
    Helper function that outputs feature weights from the trained logistic regression model.

    Arguments:
    -----------
    df : pd.dataframe, dataframe to be passed as input
    target_header : string, the column with the header description of the target label
    target_label : string, optional input if the target column is not yet encoded with binary 0 and 1 labels
    encoder : selection of 'one_hot', 'label_encoding', the type of encoding method for categorical data
    numerical_imputer : selection of 'mean', 'median', 'most_frequent', the type of imputation strategy for processing missing data
    scaler : string, selection of 'standard', 'minmax' or 'robust', type of scaler used for data processing
    reg_C : float, regularization parameter for logistic regression (inverse strength of regularization)
    reg_norm : selection of 'l1', 'l2', the type of L1 or L2 penalty for logistic regression

    Returns:
    -----------
    df_features : pd.dataframe, resulting dataframe of model feature weights as output
    """

    print('Inspecting data values... ', end='')
    # Separate features and target data
    df_y = df[[target_header]].copy()
    df_x = df.drop(columns=[target_header]).copy()
    
    # Isolate sub-dataset containing categorical values
    categorical = df_x.loc[:, df_x.dtypes == object].copy()
    
    # Isolate sub-dataset containing non-categorical values
    non_categorical = df_x.loc[:, df_x.dtypes != object].copy()
    binary_col_headers = get_binary_headers(non_categorical, non_categorical.columns.tolist())
    non_categorical.drop(columns=binary_col_headers, inplace=True)
    non_categorical_headers = non_categorical.columns.tolist()
    print('[Done]')

    # Apply numerical imputation processing to data
    if numerical_imputer != None:
        print('Imputing numerical data... ', end='')
        numerical_imputation = Imputer(strategy=numerical_imputer)
        non_categorical = numerical_imputation.fit_transform(non_categorical)
        print('[Done]')

    # Apply scaler to data
    if scaler != None:
        print('Scaling numerical data... ', end='')
        if scaler == 'standard':
            scaler = StandardScaler()
            scaler.fit(non_categorical)
        elif scaler == 'minmax':
            scaler = MinMaxScaler()
            scaler.fit(non_categorical)
        else:
            scaler = RobustScaler()
            scaler.fit(non_categorical)
        non_categorical_data = scaler.transform(non_categorical)
        non_categorical = pd.DataFrame(data=non_categorical_data, columns=non_categorical_headers)
        print('[Done]')

    # Apply encoding to categorical value data
    if encoder != None:
        print('Encoding categorical data... ', end='')
        if encoder == 'one_hot':
            categorical = pd.get_dummies(categorical)
        print('[Done]')
        
    # Join up categorical and non-categorical sub-datasets
    df_x = pd.concat([categorical, non_categorical], axis=1)
    feature_headers = df_x.columns
    X = df_x.values
    
    # Get the encoded target labels if necessary
    # Check if target labels are binary 0 and 1
    print('Inspect target data type... ', end='')
    binary_col_headers = get_binary_headers(df_y, [target_header])
    if target_header in binary_col_headers:
        y = df_y[target_header]
    else:
        # Else if column values not binary 0 and 1, proceed to encode target labels with one-hot encoding
        df_y = pd.get_dummies(df_y)

        # Select the relevant column of the specified target value as per input
        target_headers = df_y.columns.tolist()

        if target_label != None:
            for header in target_headers:
                if target_label in header:
                    y = df_y[header]
                    break
                else:
                    pass
        else:
            y = df_y.iloc[:, 0]
            print('Note: Target column contains multiple labels. \nThe column is one-hot encoded and the first column of the encoded result is selected as the target label for feature influence analysis.\n')
    print('[Done]')

    # Split train and test data for model fitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    print('Training model... no. of training examples: ' + str(X_train.shape[0]) + ', no. of features: ' + str(X_train.shape[1]) + '. ', end='')
    # Perform model training and evaluation
    model = LogisticRegression(C=reg_C, penalty=reg_norm)
    model.fit(X_train, y_train)
    print('[Done]')
    
    # Get the model performance
    y_pred = model.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    df_positive_rate = pd.DataFrame({'False positive rate' : fpr, 'True positive rate' : tpr})
    
    # Get the important features from the model in a dataframe format
    df_features = pd.DataFrame(data=model.coef_, columns=feature_headers).transpose()
    df_features.columns=['Feature weight']
    df_features.sort_values(by=['Feature weight'], ascending=False, inplace=True)
    
    print('\nLogistic Regression with ' + reg_norm.capitalize() + ' regularization model evaluation:\n')
    print(classification_report(y_test, y_pred))

    # ROC plot
    plt.figure(figsize=(8, 6))
    custom_rc = {'lines.linewidth': 0.8, 'lines.markersize': 0.8} 
    sns.set_style('white')
    sns.set_context('talk', rc=custom_rc)
    ax = sns.pointplot(x='False positive rate', y='True positive rate', data=df_positive_rate, color='Blue')
    ax.set_title('ROC plot')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')

    return df_features

# Feature analysis with random forest model
def random_forest_features(df, target_header, target_label=None, encoder=None, numerical_imputer=None, scaler=None, n_trees=10, max_depth=None, min_samples_leaf=10):
    """
    Helper function that outputs feature weights from the trained random forest model.

    Arguments:
    -----------
    df : pd.dataframe, dataframe to be passed as input
    target_header : string, the column with the header description of the target label
    target_label : string, optional input if the target column is not yet encoded with binary 0 and 1 labels
    encoder : selection of 'one_hot', 'label_encoding', the type of encoding method for categorical data
    numerical_imputer : selection of 'mean', 'median', 'most_frequent', the type of imputation strategy for processing missing data
    scaler : string, selection of 'standard', 'minmax' or 'robust', type of scaler used for data processing
    n_trees : int, number of trees/estimators for the random forest model

    Returns:
    -----------
    df_features : pd.dataframe, resulting dataframe of model feature weights as output
    """

    print('Inspecting data values... ', end='')
    # Separate features and target data
    df_y = df[[target_header]].copy()
    df_x = df.drop(columns=[target_header]).copy()
    
    # Isolate sub-dataset containing categorical values
    categorical = df_x.loc[:, df_x.dtypes == object].copy()
    
    # Isolate sub-dataset containing non-categorical values
    non_categorical = df_x.loc[:, df_x.dtypes != object].copy()
    binary_col_headers = get_binary_headers(non_categorical, non_categorical.columns.tolist())
    non_categorical.drop(columns=binary_col_headers, inplace=True)
    non_categorical_headers = non_categorical.columns.tolist()
    print('[Done]')

    # Apply numerical imputation processing to data
    if numerical_imputer != None:
        print('Imputing numerical data... ', end='')
        numerical_imputation = Imputer(strategy=numerical_imputer)
        non_categorical = numerical_imputation.fit_transform(non_categorical)
        print('[Done]')

    # Apply scaler to data
    if scaler != None:
        print('Scaling numerical data... ', end='')
        if scaler == 'standard':
            scaler = StandardScaler()
            scaler.fit(non_categorical)
        elif scaler == 'minmax':
            scaler = MinMaxScaler()
            scaler.fit(non_categorical)
        else:
            scaler = RobustScaler()
            scaler.fit(non_categorical)
        non_categorical_data = scaler.transform(non_categorical)
        non_categorical = pd.DataFrame(data=non_categorical_data, columns=non_categorical_headers)
        print('[Done]')

    # Apply encoding to categorical value data
    if encoder != None:
        print('Encoding categorical data... ', end='')
        if encoder == 'one_hot':
            categorical = pd.get_dummies(categorical)
        print('[Done]')
        
    # Join up categorical and non-categorical sub-datasets
    df_x = pd.concat([categorical, non_categorical], axis=1)
    feature_headers = df_x.columns
    X = df_x.values

    # Get the encoded target labels if necessary
    # Check if target labels are binary 0 and 1
    print('Inspecting target data type... ', end='')
    binary_col_headers = get_binary_headers(df_y, [target_header])
    if target_header in binary_col_headers:
        y = df_y[target_header]
    else:
        # Else if column values not binary 0 and 1, proceed to encode target labels with one-hot encoding
        df_y = pd.get_dummies(df_y)

        # Select the relevant column of the specified target value as per input
        target_headers = df_y.columns.tolist()

        if target_label != None:
            for header in target_headers:
                if target_label in header:
                    y = df_y[header]
                    break
                else:
                    pass
        else:
            y = df_y.iloc[:, 0]
            print('Note: Target column contains multiple labels. \nThe column is one-hot encoded and the first column of the encoded result is selected as the target label for feature influence analysis.\n')
    print('[Done]')

    # Split train and test data for model fitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
 
    print('Training model... no. of training examples: ' + str(X_train.shape[0]) + ', no. of features: ' + str(X_train.shape[1]) + '. ', end='')
    # Perform model training and evaluation
    model = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    model.fit(X_train, y_train)
    print('[Done]')
    
    # Get the model performance
    y_pred = model.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    df_positive_rate = pd.DataFrame({'False positive rate' : fpr, 'True positive rate' : tpr})
    
    # Get the important features from the model in a dataframe format
    df_features = pd.DataFrame(data=model.feature_importances_, index=feature_headers)
    df_features.columns=['Feature weight']
    df_features.sort_values(by=['Feature weight'], ascending=False, inplace=True)
    
    print('\nRandom Forest model evaluation:\n')
    print(classification_report(y_test, y_pred))

    # ROC plot
    plt.figure(figsize=(8, 6))
    custom_rc = {'lines.linewidth': 0.8, 'lines.markersize': 0.8} 
    sns.set_style('white')
    sns.set_context('talk', rc=custom_rc)
    ax = sns.pointplot(x='False positive rate', y='True positive rate', data=df_positive_rate, color='Blue')
    ax.set_title('ROC plot')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')

    return df_features

# Secondary helper function to check whether column(s) contain binary 0 and 1 values
def get_binary_headers(df, column_headers):
    binary_col_headers = []
    for header in column_headers:
        if len(df[header].unique()) == 2 and (0 in df[header].unique()) and (1 in df[header].unique()):
            binary_col_headers.append(header)
        else:
            pass
    return binary_col_headers

# Secondary helper function to get the tickers for plotting
def get_tickers(df_plot, base_interval=0.05):
    # Set the x-axis limits and marker locations at base interval increments
    x_mag = df_plot.iloc[:, 0].abs().max()*1.2
    x_mag = base_interval * (x_mag // base_interval) + base_interval
    
    n_ticks = int(2*x_mag/base_interval)
    if n_ticks > 200:
        tick_interval = base_interval*20
    elif n_ticks <= 200 and n_ticks > 100:
        tick_interval = base_interval*10
    elif n_ticks <= 100 and n_ticks > 50:
        tick_interval = base_interval*5
    elif n_ticks <= 50 and n_ticks > 15:
        tick_interval = base_interval*2
    elif n_ticks <= 15 and n_ticks > 3:
        tick_interval = base_interval
    else:
        tick_interval = base_interval*0.25

    if df_plot.iloc[:, 0].min() < 0:
        xmax = x_mag
        xmin = -x_mag
    else:
        xmax = x_mag
        xmin = 0

    xticks_range = np.linspace(xmin, xmax, int(2*x_mag/tick_interval) + 1)

    return xticks_range, xmax, xmin

# Plot the correlations of the features with respect to a target column header in the dataset.
def barplot_features(df, x_label_desc='x label', remove_zeros=True, plot_size=(10, 10), sns_style='whitegrid', sns_context='talk', sns_palette='coolwarm'):
    """
    Helper function that outputs a plot of feature correlations against a specified column in the dataset.

    Arguments:
    -----------
    df : pd.dataframe, dataframe to be passed as input
    x_label_desc : string, the x-axis label description
    remove_zeros : boolean, choose whether to remove all 0 values from feature influence plot
    plot_size : tuple, the specified size of the plot chart in the notebook cell
    sns_style : selection of builtin Seaborn set_style, background color theme categories (e.g. 'whitegrid', 'white', 'darkgrid', 'dark', etc)
    sns_context : selection of builtin Seaborn set_context, labels/lines categories (e.g. 'talk', 'paper', 'poster', etc)
    sns_palette : selection of builtin Seaborn palette, graph color theme categories (e.g. 'coolwarm', 'Blues', 'BuGn_r', etc, note adding '_r' at the end reverses the displayed color order)

    Returns:
    -----------
    Display of bar chart
    """

    # Reshape input dataframe for plotting
    if df.shape[0] > 30:
        df_plot = pd.concat([df.head(15), df.tail(15)], axis=0)
    else:
        df_plot = df

    # Remove 0 values
    if remove_zeros == True:
        df_plot = df_plot.loc[(df_plot != 0).any(axis=1)]

    # Get the tick markers range, max, and min values for plotting
    xticks_range, xmax, xmin = get_tickers(df_plot, base_interval=0.05)
    
    # Define the style of the Seaborn plot
    sns.set_style(sns_style)
    sns.set_context(sns_context)

    # Create the plot
    plt.figure(figsize=plot_size)
    plt.xticks(xticks_range)
    plt.xlim(xmin=xmin, xmax=xmax)
    ax = sns.barplot(data=df_plot, x=df_plot.columns[0], y=df_plot.index.tolist(), palette=sns_palette)
    ax.set_xlabel(x_label_desc)

# Plot the PCA features contributions chart with respect to a specified principal component index.
def barplot_features_pca(df_pca_comp, pc_index=1, x_label_desc='PC contribution', remove_zeros=True, plot_size=(10, 10), sns_style='whitegrid', sns_context='talk', sns_palette='coolwarm'):
    """
    Helper function that outputs a plot of PCA features contributions on specified a principal component.

    Arguments:
    -----------
    df_pca_comp : pd.dataframe, dataframe containing PCA features contributions to be passed as input
    pc_index : int, the index of the principal component to be extracted and plotted
    x_label_desc : string, the x-axis label description
    remove_zeros : boolean, choose whether to remove all 0 values from feature influence plot
    plot_size : tuple, the specified size of the plot chart in the notebook cell
    sns_style : selection of builtin Seaborn set_style, background color theme categories (e.g. 'whitegrid', 'white', 'darkgrid', 'dark', etc)
    sns_context : selection of builtin Seaborn set_context, labels/lines categories (e.g. 'talk', 'paper', 'poster', etc)
    sns_palette : selection of builtin Seaborn palette, graph color theme categories (e.g. 'coolwarm', 'Blues', 'BuGn_r', etc, note adding '_r' at the end reverses the displayed color order)

    Returns:
    -----------
    Display of bar chart
    """

    # Reshape input dataframe for plotting
    target_header = 'PC_' + str(pc_index)
    df = pd.DataFrame(data=df_pca_comp.iloc[pc_index - 1].sort_values(ascending=False))
    df.columns = [target_header]
    if df.shape[0] > 30:
        df_plot = pd.concat([df.head(15), df.tail(15)], axis=0)
    else:
        df_plot = df
    
    # Remove 0 values
    if remove_zeros == True:
        df_plot = df_plot.loc[(df_plot != 0).any(axis=1)]
        
    # Get the tick markers range, max, and min values for plotting
    xticks_range, xmax, xmin = get_tickers(df_plot, base_interval=0.05)

    # Define the style of the Seaborn plot
    sns.set_style(sns_style)
    sns.set_context(sns_context)

    # Create the plot
    plt.figure(figsize=plot_size)
    plt.xticks(xticks_range)
    plt.xlim(xmin=xmin, xmax=xmax)
    ax = sns.barplot(data=df_plot, x=target_header, y=df_plot.index.tolist(), palette=sns_palette)
    ax.set_xlabel(x_label_desc)

# Plot 2D scatter of PCA biplot
def biplot_pca(df_pca, target_header, pc_axes=(1, 2), sns_style='white', sns_context='talk', sns_palette='plasma'):
    """
    Produce a PCA scatter biplot.

    Arguments:
    -----------
    df_pca : pd.dataframe, PCA components dataframe as input data
    target_header : string, column header of the target label
    pc_axes : tuple, indicates the principal components to be assigned to the respective x and y axes
    sns_style : selection of builtin Seaborn set_style, background color theme categories (e.g. 'whitegrid', 'white', 'darkgrid', 'dark', etc)
    sns_context : selection of builtin Seaborn set_context, labels/lines categories (e.g. 'talk', 'paper', 'poster', etc)
    sns_palette : selection of builtin Seaborn palette, graph color theme categories (e.g. 'coolwarm', 'Blues', 'BuGn_r', etc, note adding '_r' at the end reverses the displayed color order)

    Returns:
    -----------
    Display of PCA biplot
    """
    
    # Define the style of the Seaborn plot
    sns.set_style(sns_style)
    sns.set_context(sns_context)
    
    # Create the plot
    sns.lmplot(data=df_pca, x='PC_' + str(pc_axes[0]), y='PC_' + str(pc_axes[1]), hue=target_header, fit_reg=False, palette=sns_palette, size=8, aspect=1.5)
    ax = plt.gca()
    ax.set_title('PCA biplot')
    ax.set_xlabel('Principal component ' + str(pc_axes[0]))
    ax.set_ylabel('Principal component ' + str(pc_axes[1]))

# Perform PCA and output heatmap.
def heatmap_pca(df_pca_comp, pc_max=3, sns_cmap='plasma', annot=False, plot_size=(12, 8)):
    """
    Produce a PCA heatmap after applying scaler functions using Sklearn python library.

    Arguments:
    -----------
    df_pca_comp : pd.dataframe, dataframe containing PCA components attribution as the input data
    pc_max : integer, maximum number of principal components to be displayed in the heatmap
    sns_cmap : selection of 'plasma' etc, type of color map setting for heatmap
    annot : boolean, choice of true/false for display or not display value annotations on the heatmap
    plot_size : tuple, the specified size of the plot chart in the notebook cell

    Returns:
    -----------
    df_comp : pd.dataframe, dataframe containing principal component values
    """
    # Set header descriptions for displaying PCA results
    pc_headers = ['PC_' + str(pc_index + 1) for pc_index in range(pca_max)]

    # Plot the PCA heatmap
    df_pca_comp = df_pca_comp.set_index([pc_headers])
    plt.figure(figsize=plot_size)
    g_pca_heatmap = sns.heatmap(data=df_pca_comp, annot=annot, cmap=sns_cmap)

    return df_comp

# Get dataframe that transforms/encodes discrete numbered features (e.g. 0 or 1, or 2, 10, 15) into continuous set of numbers
# Note: this adds some degree of randomisation of data, and applying encode based on the average of other samples (with exclusion
# of the active data point)
# Such feature engineering method may be useful with certain classifiers.
def get_continuous_mean(df, feature_heading, random_scaling=0.01):
    feature_series = df[feature_heading]
    feature_series_mean = [
        ((sum(feature_series) - val)/(len(feature_series) - 1)) for val in feature_series]
    random_factor = np.random.rand(len(feature_series_mean))*random_scaling + 1
    feature_series_mean = np.multiply(np.asarray(
        feature_series_mean), random_factor).tolist()
    df_continuous_mean = pd.DataFrame(
        {feature_heading: feature_series, feature_heading + '_mean_encoding': feature_series_mean})
    return df_continuous_mean

# Apply sklearn scalers and output plots
def check_stats_scalers(df, feature_heading):
    # Set scaled data headers
    original_heading = feature_heading + '_original'
    std_scaled_heading = feature_heading + '_standard_scaled'
    minmax_scaled_heading = feature_heading + '_minmax_scaled'
    robust_scaled_heading = feature_heading + '_robust_scaled'

    # Reshape the 1D input data into "transposed" column-wise array for use with sklearn scaler functions
    feature_series = df[feature_heading]
    feature_array = feature_series.values.reshape(-1, 1)

    # Fit data to scaler functions and get the scaled data after transformation
    std_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()
    robust_scaler = RobustScaler()
    std_scaled_array = std_scaler.fit_transform(feature_array)
    minmax_scaled_array = minmax_scaler.fit_transform(feature_array)
    robust_scaled_array = robust_scaler.fit_transform(feature_array)

    # Append the scaled data to a custom dataframe
    df_new = pd.DataFrame({original_heading: feature_series})
    df_new[std_scaled_heading] = std_scaled_array
    df_new[minmax_scaled_heading] = minmax_scaled_array
    df_new[robust_scaled_heading] = robust_scaled_array

    # Visualise original and scaled data distributions
    original_data = pd.Series(feature_series, name=original_heading)
    std_scaled_data = pd.Series(
        df_new[std_scaled_heading], name=std_scaled_heading)
    minmax_scaled_data = pd.Series(
        df_new[minmax_scaled_heading], name=minmax_scaled_heading)
    robust_scaled_data = pd.Series(
        df_new[robust_scaled_heading], name=robust_scaled_heading)

    fig, ax = plt.subplots(2, 2, figsize=(15, 11))
    sns.kdeplot(original_data, ax=ax[0][0], shade=True, color='b')
    sns.kdeplot(std_scaled_data, ax=ax[0][1], shade=True, color='y')
    sns.kdeplot(minmax_scaled_data, ax=ax[1][0], shade=True, color='y')
    sns.kdeplot(robust_scaled_data, ax=ax[1][1], shade=True, color='y')
    return df_new

# Apply math operation scaling and output plots
def check_math_scalers(df, feature_heading):
    """
    Apply different scaler functions to a feature column using Sklearn python library.

    Arguments:
    -----------
    df : pd.dataframe, dataframe of the input data
    feature_heading : string, header description of the feature column subjected to the scaler functions

    Returns:
    -----------
    df_new : pd.dataframe, dataframe containing math function scaled values
    """
    # Set scaled data headers
    original_heading = feature_heading + '_original'
    log_scaled_heading = feature_heading + '_log_scaled'
    sqrt_scaled_heading = feature_heading + '_sqrt_scaled'
    tanh_scaled_heading = feature_heading + '_tanh_scaled'

    # Reshape the 1D input data into "transposed" column-wise array for use with sklearn scaler functions
    feature_series = df[feature_heading]

    # Fit data to scaler functions and get the scaled data after transformation
    if np.min(feature_series.values) < 0:
        feature_array = feature_series.values - \
            np.min(feature_series.values)*(1.000001)
    elif np.min(feature_series.values) == 0:
        feature_array = feature_series.values + 0.000001
    else:
        feature_array = feature_series.values
    log_scaled_array = np.log(feature_array)
    sqrt_scaled_array = np.sqrt(feature_array)
    tanh_scaled_array = np.tanh(feature_series)

    # Append the scaled data to a custom dataframe
    df_new = pd.DataFrame({original_heading: feature_series})
    df_new[log_scaled_heading] = log_scaled_array
    df_new[sqrt_scaled_heading] = sqrt_scaled_array
    df_new[tanh_scaled_heading] = tanh_scaled_array

    # Visualise original and scaled data distributions
    original_data = pd.Series(feature_series, name=original_heading)
    log_scaled_data = pd.Series(
        df_new[log_scaled_heading], name=log_scaled_heading)
    sqrt_scaled_data = pd.Series(
        df_new[sqrt_scaled_heading], name=sqrt_scaled_heading)
    tanh_scaled_data = pd.Series(
        df_new[tanh_scaled_heading], name=tanh_scaled_heading)

    fig, ax = plt.subplots(2, 2, figsize=(15, 11))
    sns.kdeplot(original_data, ax=ax[0][0], shade=True, color='b')
    sns.kdeplot(log_scaled_data, ax=ax[0][1], shade=True, color='y')
    sns.kdeplot(sqrt_scaled_data, ax=ax[1][0], shade=True, color='y')
    sns.kdeplot(tanh_scaled_data, ax=ax[1][1], shade=True, color='y')
    return df_new

# Create dataframe for initial step of data analysis using Pandas.
def create_dataframe(file_name='unknown', dtype_dict=None):
    """
    Import original data for analysis.

    Arguments:
    -----------
    file_name : string, name of the data file (excludes extension descriptions like '.csv' or '.xlsx')
    dtype_dict : dict, dictionary to specify the variable data type(s) to be read via the Pandas read_csv() function

    Returns:
    -----------
    df_original : pd.dataframe, dataframe of the original dataset
    """

    base_dir = os.path.dirname(os.path.realpath('__file__'))
    data_dir = os.path.join(base_dir, 'Data')
    file_dir_csv = os.path.join(data_dir, file_name + '.csv')

    file_dir_xlsx = os.path.join(data_dir, file_name + '.xlsx')
    df_original = None

    try:
        df_original = pd.read_csv(file_dir_csv, encoding='cp1252', low_memory=False, dtype=dtype_dict)
        print('Status: ' + file_name + ' imported!')
    except:
        try:
            df_original = pd.read_excel(file_dir_xlsx, sheet_name='Sheet1')
            print('Status: ' + file_name + ' imported!')
        except:
            print('Status: ' + file_name + ' is unable to be read! Please ensure file content is not corrupted or if the file is of .xlsx format, the sheetname is titled as "Sheet1".')

    return df_original

# Get the file path of the input dataset - for use in the notebook template).
def get_filepath(file_name='unknown'):
    """
    Get original data's file path

    Arguments:
    -----------
    file_name : string, name of the data file (excludes extension descriptions like '.csv' or '.xlsx')

    Returns:
    -----------
    file_dir : string, directory of the data file
    """

    base_dir = os.path.dirname(os.path.realpath('__file__'))
    data_dir = os.path.join(base_dir, 'Data')
    file_dir = os.path.join(data_dir, file_name + '.csv')

    if os.path.exists(file_dir):
        pass
    else:
        file_dir = os.path.join(data_dir, file_name + '.xlsx')

    return file_dir

# Train/validation/test split function for specified data proportions of each category.
def training_split(X, y, train_partition=0.6, dev_partition=0.2, test_partition=0.2, random_state=None):
    """
    Split the data into training, validation and test portions.

    Arguments:
    -----------
    X : np.array, feature samples for training
    y : np.array, target samples for training
    train_partition : float, the proportion of the data to be used as training samples
    dev_partition : float, the proportion of the data to be used as validation samples to optimize training
    test_partition : float, the proportion of the data to be used as test samples to evaluate model performance

    Returns:
    -----------
    X_train : np.array, input training set
    y_train : np.array, target training set
    X_dev : np.array, input development set (for training optimization)
    y_dev : np.array, target development set (for training optimization)
    X_test : np.array, input test set
    y_test : np.array, target test set
    input_features : int/array, number of features in the input training/development set data
    """

    X_stats = None
    input_features = None
    partition = 0
    if train_partition + dev_partition + test_partition == 1:
        X_train_dev, X_test, y_train_dev, y_test = train_test_split(
            X, y, test_size=test_partition, random_state=random_state)
        if dev_partition != 0:
            partition = dev_partition/(1 - test_partition)
            X_train, X_dev, y_train, y_dev = train_test_split(
                X_train_dev, y_train_dev, test_size=partition, random_state=random_state)
        else:
            X_train = X_train_dev
            y_train = y_train_dev
            X_dev = []
            y_dev = []
    else:
        print('Status: Data split partitions does not add up to 1.')
        print('Using default partitions: train_parition=0.6, dev_partition=0.2, test_partition=0.2\n')
        train_partition = 0.6
        dev_partition = 0.2
        test_partition = 0.2
        X_train_dev, X_test, y_train_dev, y_test = train_test_split(
            X, y, test_size=test_partition, random_state=random_state)
        partition = dev_partition/(1 - test_partition)
        X_train, X_dev, y_train, y_dev = train_test_split(
            X_train_dev, y_train_dev, test_size=partition, random_state=random_state)

    # Get input_feature shape by checking X_train shape
    X_stats = len(X_train.shape)
    if X_stats == 2:
        input_features = X_train.shape[1]
    elif X_stats == 4:
        input_features = (X_train.shape[1], X_train.shape[2])
    else:
        pass

    print('Data partitions:')
    print('Number of training set samples: ' + str(len(X_train)))
    print('Number of development set samples: ' + str(len(X_dev)))
    print('Number of testing set samples: ' + str(len(X_test)))
    if X_stats == 2:
        print('Number of input features: ' + str(input_features))
    elif X_stats == 4:
        print('Number of input features (2D): ' +
              str(input_features[0]) + ' x ' + str(input_features[1]))
    else:
        pass

    return X_train, y_train, X_dev, y_dev, X_test, y_test, input_features


def df_to_tf(X, y):
    df_dataset = pd.concat([X, y], axis=1)
    tf_dataset = np.float32(df_dataset.values)
    tf_dataset = tf.convert_to_tensor(tf_dataset)
    return tf_dataset

# Process input image data for computer vision tasks.
def inspect_image(file_name='unknown', file_type='jpeg', grayscale=False, target_res=128):
    """
    Parse single image as numpy array. Also show image information.

    Arguments:
    -----------
    file_name : string, name of the image file (excluding extension)
    file_type : string, file type of the image
    grayscale : boolean, true/false choice of processing the image as grayscale or colour
    target_res : int, final resolution after image rescaling
    anti_aliasing : boolean, true/false choice of applying anti-aliasing if rescaling the image

    Returns:
    -----------
    img_data : np.array, parsed image data
    """

    base_dir = os.path.dirname(os.path.realpath('__file__'))
    data_dir = os.path.join(base_dir, 'Data')
    file_dir1 = ''
    file_dir2 = ''
    parse_file = True
    img_data = None
    scale_x = 0
    scale_y = 0

    if file_type == 'jpeg':
        file_dir1 = os.path.join(data_dir, file_name + '.jpg')
        file_dir2 = os.path.join(data_dir, file_name + '.jpeg')
    elif file_type == 'png':
        file_dir1 = os.path.join(data_dir, file_name + '.png')
    elif file_type == 'tiff':
        file_dir1 = os.path.join(data_dir, file_name + '.tif')
    else:
        parse_file = False
        print('Unrecognised image file type!')

    if parse_file == True:
        try:
            try:
                img_data = imread(file_dir1, as_grey=grayscale)
                if target_res != img_data.shape[0]:
                    scale_x = target_res/img_data.shape[0]
                    scale_y = target_res/img_data.shape[1]
                    img_data = rescale(img_data, scale=(scale_x, scale_y))
                print('Status: Image parsed successfully!')
                print('Image resolution: ' + str(img_data.shape[0]) + ' x ' + str(
                    img_data.shape[1]) + ', rescale factor: ' + str(scale_x) + ' x ' + str(scale_y))
                print('Number of image colour channel(s): ' +
                      str(img_data.shape[2]))
                imshow(img_data)
            except:
                try:
                    img_data = imread(file_dir2, as_grey=grayscale)
                    if target_res != img_data.shape[0]:
                        scale_x = target_res/img_data.shape[0]
                        scale_y = target_res/img_data.shape[1]
                        img_data = rescale(img_data, scale=(scale_x, scale_y))
                    print('Status: Image parsed successfully!')
                    print('Image resolution: ' + str(img_data.shape[0]) + ' x ' + str(
                        img_data.shape[1]) + ', rescale factor: ' + str(scale_x) + ' x ' + str(scale_y))
                    print('Number of image colour channel(s): ' +
                          str(img_data.shape[2]))
                    imshow(img_data)
                except Exception as e:
                    print('Status: Image inspection unsuccessful, check for error:')
                    print(e)
        except Exception as e:
            print('Status: Image parsed unsuccessfully, check for error:')
            print(e)

    return img_data

# Process all image files in the specified folder for computer vision tasks.
def parse_image_data(img_folder_names=[], file_type='jpeg', grayscale=False, target_res=128):
    """
    Process images stored in designated folders for model training.

    Arguments:
    -----------
    img_folder : string, name of the image file (excluding extension)
    file_type : string, file type of the image
    grayscale : boolean, true/false choice of processing the image as grayscale or colour
    rescale_factor : float, factor with which to rescale the image (e.g. rescaling to 1/4 of the original image is done via a factor of 0.25)

    Returns:
    -----------
    X : np.array, parsed image data as input samples
    y : np.array, target samples
    """

    base_dir = os.path.dirname(os.path.realpath('__file__'))
    data_dir = os.path.join(base_dir, 'Data')
    target_dir = ''
    parse_file = True
    scale_x = 0
    scale_y = 0
    X = []
    y = []

    if file_type == 'jpeg' or file_type == 'png' or file_type == 'tiff':
        parse_file = True
    else:
        parse_file = False
        print('Unrecognised image file type!')

    for folder_name in img_folder_names:
        if folder_name in os.listdir(data_dir):
            print('Status: Image folder "' + folder_name + '" found')
        else:
            print('Status: Image folder "' + folder_name + '" not found')
            parse_file = False

    # Parse images if folder/target label names have been setup correctly.
    if parse_file == True:
        for target in img_folder_names:
            target_dir = os.path.join(data_dir, target)
            for image in os.listdir(target_dir):
                img_data = imread(os.path.join(
                    target_dir, image), as_grey=grayscale)
                if target_res != img_data.shape[0]:
                    scale_x = target_res/img_data.shape[0]
                    scale_y = target_res/img_data.shape[1]
                    img_data = rescale(img_data, scale=(scale_x, scale_y))
                if grayscale == True:
                    img_data = img_data.reshape(
                        img_data.shape[0], img_data.shape[1], 1)
                X.append(img_data)
                y.append(target)

        # Convert image data into numpy arrays, then cale the raw pixel intensities to the range [0, 1]
        X = np.array(X)
        X = np.array(X, dtype="float")/255.0

        # Convert target label data into numpy arrays, then reshape the data to be m x 1 (where m is number of training samples)
        y = np.array(y)
        y = y.reshape(y.shape[0], 1)

        print('Images processed successfully!')
    else:
        print('Images processed unsuccessfully!')

    return X, y
