# This module is made to aid data analysis by providing commonly used functions
# for inspecting and analysing a dataset.

# Setup data processing dependencies
import time
import os
import os.path
import gc
import warnings
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
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, roc_curve, precision_recall_curve, roc_auc_score, auc
from skimage.io import imread, imshow
from skimage.transform import rescale, resize, downscale_local_mean
from datetime import datetime, date
try:
    from google.colab import files
except:
    pass
warnings.simplefilter(action='ignore', category=FutureWarning)

# ===============================
# === Inspection and analysis ===

# Jupyter notebook styling - centering function
# from IPython.core.display import HTML
# HTML("""
# <style>
# .output_png {
#     display: table-cell;
#     text-align: center;
#     vertical-align: middle;
# }
# </style>
# """)

# Secondary helper function to get a list of all files within a directory tree (for file search)
def scan_dir(dir, file_list=[]):
    # Recursively search through the directory tree and append file name and file path 
    try:
        # For native directory files
        for name in os.listdir(dir):
            path = os.path.join(dir, name)
            if os.path.isfile(path):
                file_list.append((name, path))
            else:
                scan_dir(path, file_list)
    except:
        try:
            # For Google Drive files
            dir = os.path.join(dir, 'drive')
            for name in files.os.listdir(dir):
                path = os.path.join(dir, name)
                if os.path.isfile(path):
                    file_list.append((name, path))
                else:
                    scan_dir(path, file_list)
        except:
            pass

    return file_list

# Secondary helper function for file search
def locate_file(file_name):
    base_dir = os.path.dirname(os.path.realpath('__file__'))
    
    file_name_base = file_name.lower()
    original_file_name = None
    file_dir = None

    # Get a list containing all the files within the base directory tree
    file_list = scan_dir(base_dir)

    # Look for the specific file in the file list that matches the required file name
    for f_name, f_dir in file_list:
        if f_name.endswith('.csv'):
            f_name_base = f_name.replace('.csv', '').lower()
            if file_name_base == f_name_base:
                original_file_name = f_name
                file_dir = f_dir
                break
        if f_name.endswith('.xlsx'):
            f_name_base = f_name.replace('.xlsx', '').lower()
            if file_name_base == f_name_base:
                original_file_name = f_name
                file_dir = f_dir
                break
        if f_name.endswith('.xls'):
            f_name_base = f_name.replace('.xls', '').lower()
            if file_name_base == f_name_base:
                original_file_name = f_name
                file_dir = f_dir
                break

    return original_file_name, file_dir

# Read raw data into Pandas dataframe for analysis.
def read_data(file_name, encoding='utf-8', sheet_name='Sheet1'):
    """
    Read data into dataframe for analysis.

    Arguments:
    -----------
    file_name : string, name of the data file (excludes extension descriptions e.g. '.csv', '.xlsx' or 'xls')
    sheet_name : string, name of the Excel sheet containing the data to be read

    Returns:
    -----------
    df_read : pd.dataframe, the dataframe read from the dataset
    """
    # Get the directory of the data file
    original_file_name, file_dir = locate_file(file_name)

    # Read the file content into a dataframe
    df_read = None
    if file_dir is not None:
        if file_dir.endswith('.csv'):
            try:
                df_read = pd.read_csv(file_dir, encoding=encoding)
                print('Status: "' + original_file_name + '" has been successfully read into dataframe!')
            except:
                print('Status: "' + original_file_name + '" cannot be read into dataframe. Note: CSV file format is detected.')
                raise
        elif file_dir.endswith('.xlsx'):
            try:
                df_read = pd.read_excel(file_dir, sheet_name=sheet_name)
                print('Status: "' + original_file_name + '" has been successfully read into dataframe!')
            except:
                print('Status: "' + original_file_name + '" cannot be read into dataframe. Note: Excel file format is detected (ensure sheetname of the content is titled "sheet1".')
                raise
        elif file_dir.endswith('.xls'):
            try:
                df_read = pd.read_excel(file_dir, sheet_name=sheet_name)
                print('Status: "' + original_file_name + '" has been successfully read into dataframe!')
            except:
                print('Status: "' + original_file_name + '" cannot be read into dataframe. Note: Excel file format is detected (ensure sheetname of the content is titled "sheet1".')
                raise
    else:
        print('Status: "' + file_name + '" cannot be located within the current file directory (and its sub-directories)')

    return df_read

# Get high level information about the imported datasets
def data_overview(df_list):
    '''
    Display high level information about the datasets (including memory, number of rows/columns).

    df_list : list of pd.dataframes as input

    Returns:
    -----------
    df_summary : pd.dataframe, display of data overview information
    '''
    df_name = []
    df_mem = []
    df_rows = []
    df_columns = []
    for df in df_list:
        nrows = df.shape[0]
        ncols= df.shape[1]
        mem_used = df.memory_usage(index=True).sum()/(10**9)
        mem_used = round(mem_used, 6)
        df_mem.append(mem_used)
        df_name.append('DF ' + str(len(df_name) + 1))
        df_rows.append(nrows)
        df_columns.append(ncols)
        
    df_summary = pd.DataFrame({'Dataframe' : df_name, 'Memory usage (GB)' : df_mem, 'No. rows' : df_rows, 'No. columns' : df_columns})
    df_summary = df_summary[['Dataframe', 'Memory usage (GB)', 'No. rows', 'No. columns']]
    df_summary.set_index('Dataframe', inplace=True)

    return df_summary

# Join multiple dataframes along rows or columns.
def join_dataframes(df_list, axis=0):
    '''
    Joins subset dataframes into one output dataframe (subset dataframe is removed from memory).

    df_list : list of pd.dataframes as input
    axis : 0 or 1, 0 - joining on rows, 1 - joining on columns

    Returns:
    -----------
    df_output : pd.dataframe as output
    dfs_cleared : list of pd.dataframes that are cleared
    '''
    df_output = pd.DataFrame()
    df_cleared = []
    i = 0
    for df in df_list:
        if i == 0:
            df_output = df
            # Clear original dataframe from memory
            dfs_cleared.append(df.iloc[0:0])
            print('Status: dataframe #' + str(i + 1) + ' processed!')
        else:
            df_output = pd.concat([df_output, df], axis=axis)
            # Clear original dataframe from memory
            df_cleared.append(df.iloc[0:0])
            print('Status: dataframe #' + str(i + 1) + ' processed!')

        i += 1
    
    gc.collect()

    return df_output, df_cleared

# Get the words and word count info overview
def word_count(df_series, top_count=None, min_count=None, max_count=None):
    """
    Helper function to determine the words and the respective counts of the words.
    Arguments:
    -----------
    df_series : pd.series, pandas dataframe series containing the groups of words
    top_count : int, the top most common words and counts
    min_count : int, filtering of greater than or equal to the minimum specified word count in the result
    max_count : int, filtering of less than or equal to the maximum specified word count in the result
    
    Returns:
    -----------
    df_word_count : pd.dataframe, the dataframe output as result
    """
    
    # Splits on whitespace
    df_words = [str(words).split() for words in df_series]
    # Flatten text groups into a flat list of words
    df_words_flat = [word for group in df_words for word in group]
    
    # Get top common words and counts
    common_words = [word for word, count in Counter(df_words_flat).most_common(top_count)]
    common_word_count = [count for word, count in Counter(df_words_flat).most_common(top_count)]
    df_word_count = pd.DataFrame({'Word' : common_words, 'Word count' : common_word_count})
    
    # Filter further by min or max word count parameters
    if min_count is not None:
        df_word_count = df_word_count[df_word_count['Word count'] >= min_count]
    if max_count is not None:
        df_word_count = df_word_count[df_word_count['Word count'] <= max_count]
    
    return df_word_count

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
def missing_values(df):
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
    if mis_val_table_ren_columns.shape[0] != 0:
        return mis_val_table_ren_columns

# Secondary helper function for creating custom (binary) target labels.
def target_label(row, target_header, lookup_value, criteria, pos_label, neg_label):
    value = row[target_header]
    output = neg_label
    if isinstance(criteria, (list, tuple)):
        for c in criteria:
            if 'greater than' in c and isinstance(value, (float, int)):
                if value > lookup_value:
                    output = pos_label 

            if 'less than' in c and isinstance(value, (float, int)):
                if value < lookup_value:
                    output = pos_label

            if 'equal to' in c:
                if value == lookup_value:
                    output = pos_label

            if 'contains' in c and isintance(value, str):
                if lookup_value in value:
                    output = pos_label
    else:
        if 'greater than' in criteria and isinstance(value, (float, int)):
            if value > lookup_value:
                output = pos_label

        if 'less than' in criteria and isinstance(value, (float, int)):
            if value < lookup_value:
                output = pos_label

        if 'equal to' in criteria:
            if value == lookup_value:
                output = pos_label

        if 'contains' in criteria and isintance(value, str):
            if lookup_value in value:
                output = pos_label
    
    row[target_header + '_' + str(lookup_value)] = output
    
    return row

# Helper function to creating new target labels
def create_target(df, target_header, lookup_value, criteria='equal to', pos_label=1, neg_label=0):
    """
    Helper function that outputs a table containing the newly created target label(s).

    Arguments:
    -----------
    df : pd.dataframe, dataframe to be passed as input
    target_header : string, the column with the header description to run feature correlations against
    lookup_value : the target value to be sought after in the existing target column
    criteria : string or list/tuple of string elements, selection of 'equal to', 'greater than', 'less than', 'contains', the type of operator setting
    pos_label : int, output for positive label
    neg_label : int, output for negative label

    Returns:
    -----------
    df_output : pd.dataframe, resulting dataframe as output
    """
    df_output = df.apply(lambda row : target_label(row, target_header=target_header, lookup_value=lookup_value, 
            criteria=criteria, pos_label=pos_label, neg_label=neg_label), axis=1)
    
    return df_output

# Secondary helper function for cleaning datetime data of 'datetime64ns' data type
def format_datetime64ns(row, feature_header, cleaned_header, original_format, target_format, dtime_unit, dtime_ref):
    try:
        time_sample_obj = row[feature_header]

        # Save the formatted datetime data in the specified "cleaned header" column
        row[cleaned_header] = time_sample_obj.strftime(target_format)

        # Calculate and save the delta time data in an additional column
        time_sample = time_sample_obj.date()
        if dtime_unit == 'days':
            row[cleaned_header + ' (' + dtime_unit + ' since)'] = (time_sample - dtime_ref).days
        elif dtime_unit == 'weeks':
            row[cleaned_header + ' (' + dtime_unit + ' since)'] = (time_sample - dtime_ref).weeks
    except:
        pass
    
    return row

# Secondary helper function for cleaning datetime data of 'object' data type
def format_datetimeobject(row, feature_header, cleaned_header, original_format, target_format, dtime_unit, dtime_ref):
    try:
        time_sample_obj = datetime.strptime(row[feature_header], original_format)

        # Save the formatted datetime data in the specified "cleaned header" column
        row[cleaned_header] = time_sample_obj.strftime(target_format)

        # Calculate and save the delta time data in an additional column
        time_sample = time_sample_obj.date()
        if dtime_unit == 'days':
            row[cleaned_header + ' (' + dtime_unit + ' since)'] = (time_sample - dtime_ref).days
        elif dtime_unit == 'weeks':
            row[cleaned_header + ' (' + dtime_unit + ' since)'] = (time_sample - dtime_ref).weeks
    except:
        pass
    
    return row

# Helper function to clean datetime data in the desired format
def cleaned_datetime(df, feature_header, cleaned_header=None, original_format=None, target_format=None, dtime_unit='days', dtime_ref=None):
    '''
    Helper function that produces the cleaned datetime data and the number of days/weeks time delta information in the dataset

    Arguments:
    -----------
    df : pd.dataframe, dataframe to be passed as input
    feature_header : string, the column with the header description that contains the original datetime data required for cleaned
    cleaned_header : string, the column with header description that is to contain the cleaned datetime data
    original_format : string, the datetie format of the original datetime data
    target_format : string, the datetime format specification of the output of the cleaned datetime data
    dtime_unit : selection of 'days', 'weeks' etc, the time difference variable quantity w.r.t. a set reference time
    dtime_ref : date object, a set reference time for calculating time difference

    Returns:
    -----------
    df_output : pd.dataframe, resulting dataframe as output
    '''
    print('Inspecting date/time data in column "' + feature_header + '"... ', end='')
    df_output = df

    if original_format is None:
        print('[Done]')
        print('Status: Original datetime format "original_format" is not specified. Function is terminated.')
        print('Please state the "original_format" as per examples:')
        print('Example 1: "2018-Mar-01", format: "%Y-%b-%d"')
        print('Example 2: "2018-03-01", format: "%Y-%m-%d"')
        print('Example 3: "01-Mar-2018", format: "%d-%b-%Y"')
        print('Example 4: "01-03-2018", format: "%d-%m-%Y"')
    else:
        print('[Done]')
        if cleaned_header is None:
            cleaned_header = feature_header
            print('Status: Input of "cleaned_header" is unspecified. The original datetime column will be overwritten with the newly cleaned data.')

        if target_format is None:
            target_format = '%Y-%m-%d'
            print('Status: Input of "target_format" is unspecified. The format of the newly cleaned data will be set to YYYY-MM-DD.')
        
        if dtime_ref is None:
            dtime_ref = date(1999, 12, 31)
            print('Status: Input of "dtime_ref" (reference for delta time measurement) is unspecified. Setting reference time to "date(1999, 12, 31)".')
     
        # Get the original datetime format of the data
        original_dtype = df.dtypes[feature_header].name
        print('Status: The data type of the original datetime content is "' + str(original_dtype) + '".')

        # Process datetime data to the desired format (additionally, process delta time contents)
        print('Cleaning date/time data and update dataset... ', end='')
        if str(original_dtype) == 'datetime64[ns]':
            df_output = df.apply(lambda row : format_datetime64ns(row, feature_header, cleaned_header, original_format, target_format, dtime_unit, dtime_ref), axis=1)
            df_output[cleaned_header + ' (' + dtime_unit + ' since)'] = df_output[cleaned_header + ' (' + dtime_unit + ' since)'].astype(float)
            print('[Done]')
        elif str(original_dtype) == 'object':
            df_output = df.apply(lambda row : format_datetimeobject(row, feature_header, cleaned_header, original_format, target_format, dtime_unit, dtime_ref), axis=1)
            df_output[cleaned_header + ' (' + dtime_unit + ' since)'] = df_output[cleaned_header + ' (' + dtime_unit + ' since)'].astype(float)
            print('[Done]')
        else:
            print('[Done]')
            print('Status: Original datetime content data type is not covered by this function. Function is terminated.')

    return df_output

# Secondary helper function for labelling duplicate count index information
def value_count(value, value_lookup):
    # Lookup value against a table to check duplicated value count, increment if already exists, or initialise count index at 1.
    if value in value_lookup:
        value_lookup[value] += 1
    else:
        value_lookup[value] = 1
    
    # Get the count index of the value
    count_index = value_lookup[value]

    return count_index, value_lookup

# Helper function for labelling duplicate count index information
def label_duplicates(df, feature_header, duplicate_position=None):
    '''
    Helper function that produces an updated dataset containing the count index of duplicated values in the selected column.

    Arguments:
    -----------
    df : pd.dataframe, dataframe to be passed as input
    feature_header : string, the column with the header description that will be inspected for the count index information of duplicated values
    duplicate_position : selection of 'last', 'first', the additional information of first or last description tagged to the count index

    Returns:
    -----------
    df_output : pd.dataframe, resulting dataframe as output

    '''
    value_lookup = {}
    duplicate_index_list = []
    duplicate_position_list = []
    df_output = df

    print('Inspecting duplicate data in column "' + feature_header + '"... ', end='')
    # Get the unique value count dataframe
    df_unique_values = unique_values(df, feature_header)
    value_unique_list = df_unique_values.iloc[:, 0].values.tolist()
    num_unique_list = df_unique_values.iloc[:, 1].values.tolist()
    print('[Done]')
    
    print('Append description label(s) to duplicate data... ', end='')
    # Iterate through the data of the selected column and lookup selected values against a dictionary containing values and their count index
    for _, value in df[feature_header].iteritems():
        count_index, value_lookup = value_count(value, value_lookup)

        # Get the duplicated count numbering order of the value label
        duplicate_index_list.append(count_index)

        # Get the duplicated count numbering description tag of the value label
        if duplicate_position == 'first':
            if count_index == 1:
                duplicate_position_list.append('first')
            else:
                duplicate_position_list.append(np.nan)
        elif duplicate_position == 'last':
            max_count_index = value_unique_list.index(value)
            if count_index == num_unique_list[max_count_index]:
                duplicate_position_list.append('last')
            else:
                duplicate_position_list.append(np.nan)
        else:
            pass
    
    # Add the count index, and count index tag information as columns in the output dataset
    df_output[feature_header + ' (duplicate #)'] = pd.Series(duplicate_index_list)
    df_output[feature_header + ' (duplicate position)'] = pd.Series(duplicate_position_list)

    print('[Done]')

    return df_output

# Secondary helper function for categorical value encoding, numerical imputation and scaling
def transform_data(df, target_header, numerical_imputer, scaler, encoder, remove_binary):
    print('Inspecting data values... ', end='')
    # Separate features and target data
    df_y = df[[target_header]].copy()
    df_x = df.drop(columns=[target_header]).copy()
    
    # Isolate sub-dataset containing categorical values
    categorical = df_x.loc[:, df_x.dtypes == object].copy()
    
    # Isolate sub-dataset containing non-categorical values
    non_categorical = df_x.loc[:, df_x.dtypes != object].copy()
    non_categorical = non_categorical.astype(np.float64)
    
    if remove_binary:
        if non_categorical.shape[1] > 0:
            binary_col_headers = get_binary_headers(non_categorical, non_categorical.columns.tolist())
            non_categorical.drop(columns=binary_col_headers, inplace=True)
        print('[Done]')

    non_categorical_headers = non_categorical.columns.tolist()

    # Apply numerical imputation processing to data
    if numerical_imputer is not None:
        if non_categorical.shape[1] > 0:
            print('Imputing numerical data... ', end='')
            numerical_imputation = Imputer(strategy=numerical_imputer)
            non_categorical_data = numerical_imputation.fit_transform(non_categorical)
            non_categorical = pd.DataFrame(data=non_categorical_data, columns=non_categorical_headers)
            print('[Done]')

    # Apply scaler to data
    if scaler is not None:
        if non_categorical.shape[1] > 0:
            print('Scaling numerical data... ', end='')
            if scaler == 'standard':
                scaler_function = StandardScaler()
                scaler_function.fit(non_categorical)
            elif scaler == 'minmax':
                scaler_function = MinMaxScaler()
                scaler_function.fit(non_categorical)
            else:
                scaler_function = RobustScaler()
                scaler_function.fit(non_categorical)
            non_categorical_data = scaler_function.transform(non_categorical)
            non_categorical = pd.DataFrame(data=non_categorical_data, columns=non_categorical_headers)
            print('[Done]')

    # Apply encoding to categorical value data
    if encoder is not None:
        if categorical.shape[1] > 0:
            print('Encoding categorical data... ', end='')
            if encoder == 'one_hot':
                categorical = pd.get_dummies(categorical)
            print('[Done]')
            
    # Join up categorical and non-categorical sub-datasets
    if non_categorical.shape[1] > 0 and categorical.shape[1] > 0:
        df_x = pd.concat([non_categorical, categorical], axis=1)
    elif non_categorical.shape[1] == 0:
        df_x = categorical
    else:
        df_x = non_categorical

    return df_x, df_y

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
    main_label = target_header

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
    all_categorical_headers = df.loc[:, df.dtypes == object].columns.tolist()
    if target_header in all_categorical_headers:
        if len(df_y[target_header].unique()) == 2 and (0 in df_y[target_header].unique()) and (1 in df_y[target_header].unique()):
            pass
        else:
            # Else if column values not binary 0 and 1, proceed to encode target labels with one-hot encoding
            df_y = pd.get_dummies(df_y)

            # Select the relevant column of the specified target value as per input
            target_headers = df_y.columns.tolist()

            if target_label is not None:
                target_header = target_header + '_' + target_label
            else:
                target_header = target_headers[0]
                print('Note: Target column contains multiple labels. \nThe column is one-hot encoded and the first column of the encoded result is selected as the target label for feature influence analysis.\n')
    print('[Done]')

    print('Extracting data correlations... ', end='')
    # Get the correlation values with respect to the target column
    df_correlations = pd.DataFrame(df_x.corr()[target_header].sort_values(ascending=False))
    print('[Done]')

    # Drop the row with the index containing the original target header (i.e. drop the target label columns as correlation is relevant only for indepdent variables)
    index_labels = df_correlations.index.tolist()
    for label in index_labels:
        if main_label in label:
            df_correlations.drop(df_correlations.index[df_correlations.index.get_loc(label)], inplace=True)

    # Drop rows containing NaN
    df_correlations.dropna(inplace=True)

    return df_correlations

# Get the dataframe containing the top specified correlations
def correlated_features(df, corr, n_upper=5, n_lower=5):
    """
    Produces a subset dataframe containing the top specified correlation features

    Arguments:
    -----------
    df : pd.dataframe, dataframe to be passed as input
    corr : pd.dataframe, the custom dataframe containing preprocessed correlation values as input
    upper_corr : integer, the specified number of upper correlation features to be retained
    lower_corr : integer, the specified number of lower correlation features to be retained

    Returns:
    -----------
    df_features : pd.dataframe, resulting dataframe as output
    """

    # Get the upper and lower correlation features from the dataset
    upper_corr_features = corr.head(n_upper).index.tolist()
    lower_corr_features = corr.tail(n_lower).index.tolist()

    # Obtain output dataset containing just the specified top correlated features
    corr_features = upper_corr_features + lower_corr_features
    df_features = df[corr_features]

    return df_features

# Feature analysis with PCA
def pca_check(df, target_header, pca_components=10, numerical_imputer=None, scaler=None, encoder=None, remove_binary=False):
    """
    Helper function that outputs PCA transformation and the associated features contributions. Also outputs explained variance attributes.

    Arguments:
    -----------
    df : pd.dataframe, dataframe to be passed as input
    target_header : string, the column with the header description of the target label
    pca_components : int, the number of principal components to be extracted from PCA
    numerical_imputer : selection of 'mean', 'median', 'most_frequent', the type of imputation strategy for processing missing data
    scaler : string, selection of 'standard', 'minmax' or 'robust', type of scaler used for data processing
    encoder : selection of 'one_hot', 'label_encoding', the type of encoding method for categorical data
    remove_binary : boolean, option to remove columns containing binary values

    Returns:
    -----------
    df_pca : pd.dataframe, resulting dataframe of PCA transformed data as output
    df_pca_comp : pd.dataframe, resulting dataframe of PCA associated features contributions as output
    Display of explained variance plot
    """

    # Apply the optional data transformation (imputing, scaling, encoding) if required 
    df_x, df_y = transform_data(df, target_header, numerical_imputer, scaler, encoder, remove_binary)
        
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
    
    # Explained variance plot
    df_explained_var['PC']= [pc_index + 1 for pc_index in range(pca_components)]

    custom_rc = {'lines.linewidth': 0.8, 'lines.markersize': 0.8} 
    sns.set_style('white')
    sns.set_context('talk', rc=custom_rc)
    plt.figure(figsize=(9, 7))
    ax = sns.pointplot(x='PC', y='Explained variance %', data=df_explained_var, color='Blue')
    ax = sns.pointplot(x='PC', y='Explained variance (cumulative) %', data=df_explained_var, color='Grey')
    ax.set_title('Explained variance')
    ax.set_xlabel('Principal component')
    ax.set_ylabel('Explained variance %')
    ax.legend(labels=('Variance', 'Variance cumulative'))
    leg = ax.get_legend()
    leg.legendHandles[0].set_color('Blue')
    leg.legendHandles[1].set_color('Grey')
    
    return df_pca, df_pca_comp

# Feature analysis with logistic regression
def svm_anomaly_features(df, target_header, target_label=None, nu=0.2, numerical_imputer=None, scaler=None, encoder=None, remove_binary=False):
    """
    Helper function that outputs feature weights from the trained one-class SVM model (with linear kernel).

    Arguments:
    -----------
    df : pd.dataframe, dataframe to be passed as input
    target_header : string, the column with the header description of the target label
    target_label : string, optional input if the target column is not yet encoded with binary 0 and 1 labels
    nu : float, regularization parameter for one-class SVM where it is upper bounded at the fraction of outliers and a lower bounded at the fraction of support vectors
    numerical_imputer : selection of 'mean', 'median', 'most_frequent', the type of imputation strategy for processing missing data
    scaler : string, selection of 'standard', 'minmax' or 'robust', type of scaler used for data processing
    encoder : selection of 'one_hot', 'label_encoding', the type of encoding method for categorical data
    remove_binary : boolean, option to remove columns containing binary values
    
    Returns:
    -----------
    df_features : pd.dataframe, resulting dataframe of model feature weights as output
    """

    # Apply the optional data transformation (imputing, scaling, encoding) if required 
    df_x, df_y = transform_data(df, target_header, numerical_imputer, scaler, encoder, remove_binary)

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

        if target_label is not None:
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
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    try:
        roc_auc = auc(fpr, tpr)
        roc_auc = round(roc_auc, 2)
    except:
        roc_auc = 'undefined'
    roc_auc_label = 'ROC curve (area: ' + str(roc_auc) + ')'

    # Get the important features from the model in a dataframe format
    df_features = pd.DataFrame(data=model.coef_, columns=feature_headers).transpose()
    df_features.columns=['Feature weight']
    df_features.sort_values(by=['Feature weight'], ascending=False, inplace=True)
    
    print('\nOne-class SVM model with linear kernel evaluation:\n')
    print(classification_report(y_test, y_pred))

    # ROC plot
    plt.figure(figsize=(9, 7))
    plt.plot(fpr, tpr, color='darkblue', lw=2, label=roc_auc_label)
    plt.plot([0, 1], [0, 1], color='skyblue', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

    return df_features

# Feature analysis with logistic regression
def logistic_reg_features(df, target_header, target_label=None, reg_C=10, reg_norm='l2', numerical_imputer=None, scaler=None, encoder=None, remove_binary=False):
    """
    Helper function that outputs feature weights from the trained logistic regression model.

    Arguments:
    -----------
    df : pd.dataframe, dataframe to be passed as input
    target_header : string, the column with the header description of the target label
    target_label : string, optional input if the target column is not yet encoded with binary 0 and 1 labels
    reg_C : float, regularization parameter for logistic regression (inverse strength of regularization)
    reg_norm : selection of 'l1', 'l2', the type of L1 or L2 penalty for logistic regression
    numerical_imputer : selection of 'mean', 'median', 'most_frequent', the type of imputation strategy for processing missing data
    scaler : string, selection of 'standard', 'minmax' or 'robust', type of scaler used for data processing
    encoder : selection of 'one_hot', 'label_encoding', the type of encoding method for categorical data
    
    Returns:
    -----------
    df_features : pd.dataframe, resulting dataframe of model feature weights as output
    """

    # Apply the optional data transformation (imputing, scaling, encoding) if required 
    df_x, df_y = transform_data(df, target_header, numerical_imputer, scaler, encoder, remove_binary)

    feature_headers = df_x.columns
    X = df_x.values
    
    # Get the encoded target labels if necessary
    # Check if target labels are binary 0 and 1
    print('Inspect target data type... ', end='')
    all_categorical_headers = df.loc[:, df.dtypes == object].columns.tolist()
    if target_header in all_categorical_headers:
        binary_col_headers = get_binary_headers(df_y, [target_header])
        if target_header in binary_col_headers:
            y = df_y[target_header]
        else:
            # Else if column values not binary 0 and 1, proceed to encode target labels with one-hot encoding
            df_y = pd.get_dummies(df_y)

            # Select the relevant column of the specified target value as per input
            target_headers = df_y.columns.tolist()

            if target_label is not None:
                for header in target_headers:
                    if target_label in header:
                        y = df_y[header]
                        break
                    else:
                        pass
            else:
                y = df_y.iloc[:, 0]
                print('Note: Target column contains multiple labels. \nThe column is one-hot encoded and the first column of the encoded result is selected as the target label for feature influence analysis. ')
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
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    try:
        roc_auc = auc(fpr, tpr)
        roc_auc = round(roc_auc, 2)
    except:
        roc_auc = 'undefined'
    roc_auc_label = 'ROC curve (area: ' + str(roc_auc) + ')'
    
    # Get the important features from the model in a dataframe format
    df_features = pd.DataFrame(data=model.coef_, columns=feature_headers).transpose()
    df_features.columns=['Feature weight']
    df_features.sort_values(by=['Feature weight'], ascending=False, inplace=True)
    
    print('\nLogistic Regression with ' + reg_norm.capitalize() + ' regularization model evaluation:\n')
    print(classification_report(y_test, y_pred))

    plt.figure(figsize=(9, 7))
    plt.plot(fpr, tpr, color='darkblue', lw=2, label=roc_auc_label)
    plt.plot([0, 1], [0, 1], color='skyblue', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
    return df_features

# Feature analysis with random forest model
def random_forest_features(df, target_header, target_label=None, n_trees=10, max_depth=None, min_samples_leaf=10, numerical_imputer=None, scaler=None, encoder=None, remove_binary=False):
    """
    Helper function that outputs feature weights from the trained random forest model.

    Arguments:
    -----------
    df : pd.dataframe, dataframe to be passed as input
    target_header : string, the column with the header description of the target label
    target_label : string, optional input if the target column is not yet encoded with binary 0 and 1 labels
    n_trees : int, number of trees/estimators for the random forest model
    max_depth : int, the depth parameter of the random forest model (max level of segments in the tree model)
    min_samples_leaf : int, the min samples parameter of the random forest model (min samples required for considering decision split)
    numerical_imputer : selection of 'mean', 'median', 'most_frequent', the type of imputation strategy for processing missing data
    scaler : string, selection of 'standard', 'minmax' or 'robust', type of scaler used for data processing
    encoder : selection of 'one_hot', 'label_encoding', the type of encoding method for categorical data
    remove_binary : boolean, option to remove columns containing binary values
    
    Returns:
    -----------
    df_features : pd.dataframe, resulting dataframe of model feature weights as output
    """

    # Apply the optional data transformation (imputing, scaling, encoding) if required 
    df_x, df_y = transform_data(df, target_header, numerical_imputer, scaler, encoder, remove_binary)

    feature_headers = df_x.columns
    X = df_x.values

    # Get the encoded target labels if necessary
    # Check if target labels are binary 0 and 1
    print('Inspecting target data type... ', end='')
    all_categorical_headers = df.loc[:, df.dtypes == object].columns.tolist()
    if target_header in all_categorical_headers:
        binary_col_headers = get_binary_headers(df_y, [target_header])
        if target_header in binary_col_headers:
            y = df_y[target_header]
        else:
            # Else if column values not binary 0 and 1, proceed to encode target labels with one-hot encoding
            df_y = pd.get_dummies(df_y)

            # Select the relevant column of the specified target value as per input
            target_headers = df_y.columns.tolist()

            if target_label is not None:
                for header in target_headers:
                    if target_label in header:
                        y = df_y[header]
                        break
                    else:
                        pass
            else:
                y = df_y.iloc[:, 0]
                print('Note: Target column contains multiple labels. \nThe column is one-hot encoded and the first column of the encoded result is selected as the target label for feature influence analysis. ')
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
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    try:
        roc_auc = auc(fpr, tpr)
        roc_auc = round(roc_auc, 2)
    except:
        roc_auc = 'undefined'
    roc_auc_label = 'ROC curve (area: ' + str(roc_auc) + ')'
    
    # Get the important features from the model in a dataframe format
    df_features = pd.DataFrame(data=model.feature_importances_, index=feature_headers)
    df_features.columns=['Feature weight']
    df_features.sort_values(by=['Feature weight'], ascending=False, inplace=True)
    
    print('\nRandom Forest model evaluation:\n')
    print(classification_report(y_test, y_pred))

    # ROC plot
    plt.figure(figsize=(9, 7))
    plt.plot(fpr, tpr, color='darkblue', lw=2, label=roc_auc_label)
    plt.plot([0, 1], [0, 1], color='skyblue', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

    return df_features

# Secondary helper function for matplotlib plot
def set_theme(theme_style):
    theme = {}
    if theme_style == 'red':
        theme = {
            'facecolor' : 'salmon',
            'color' : 'white',
            'edgecolor' : 'orangered',
            'linewidth' : 1.5,
            'linestyle' : '-',
            'alpha' : 0.5
        }
        
    if theme_style == 'darkred':
        theme = {
            'facecolor' : 'red',
            'color' : 'white',
            'edgecolor' : 'firebrick',
            'linewidth' : 1.5,
            'linestyle' : '-',
            'alpha' : 0.5
        }
        
    elif theme_style == 'blue':
        theme = {
            'facecolor' : 'royalblue',
            'color' : 'white',
            'edgecolor' : 'blue',
            'linewidth' : 1.5,
            'linestyle' : '-',
            'alpha' : 0.5
        }
        
    elif theme_style == 'darkblue':
        theme = {
            'facecolor' : 'blue',
            'color' : 'white',
            'edgecolor' : 'navy',
            'linewidth' : 1.5,
            'linestyle' : '-',
            'alpha' : 0.5
        }
        
    elif theme_style == 'green':
        theme = {
            'facecolor' : 'forestgreen',
            'color' : 'white',
            'edgecolor' : 'darkgreen',
            'linewidth' : 1.5,
            'linestyle' : '-',
            'alpha' : 0.5
        }
        
    elif theme_style == 'darkgreen':
        theme = {
            'facecolor' : 'green',
            'color' : 'white',
            'edgecolor' : 'darkgreen',
            'linewidth' : 1.5,
            'linestyle' : '-',
            'alpha' : 0.5
        }
        
    elif theme_style == 'gray':
        theme = {
            'facecolor' : 'darkgray',
            'color' : 'white',
            'edgecolor' : 'dimgray',
            'linewidth' : 1.5,
            'linestyle' : '-',
            'alpha' : 0.5
        }
        
    elif theme_style == 'darkgray':
        theme = {
            'facecolor' : 'gray',
            'color' : 'white',
            'edgecolor' : 'black',
            'linewidth' : 1.5,
            'linestyle' : '-',
            'alpha' : 0.5
        }
        
    elif theme_style == 'brown':
        theme = {
            'facecolor' : 'peru',
            'color' : 'white',
            'edgecolor' : 'sienna',
            'linewidth' : 1.5,
            'linestyle' : '-',
            'alpha' : 0.5
        }
        
    elif theme_style == 'darkbrown':
        theme = {
            'facecolor' : 'brown',
            'color' : 'white',
            'edgecolor' : 'maroon',
            'linewidth' : 1.5,
            'linestyle' : '-',
            'alpha' : 0.5
        }
        
    elif theme_style == 'purple':
        theme = {
            'facecolor' : 'mediumpurple',
            'color' : 'white',
            'edgecolor' : 'darkviolet',
            'linewidth' : 1.5,
            'linestyle' : '-',
            'alpha' : 0.5
        }
        
    elif theme_style == 'darkpurple':
        theme = {
            'facecolor' : 'blueviolet',
            'color' : 'white',
            'edgecolor' : 'rebeccapurple',
            'linewidth' : 1.5,
            'linestyle' : '-',
            'alpha' : 0.5
        }

    return theme

# Secondary helper function for matplotlib fonts
def set_fonts():
    title_fonts = {
        'fontname' : 'sans-serif',
        'fontweight' : 550, 
        'fontsize' : 'xx-large',
        'fontstyle' : 'normal',
        'fontvariant' : 'normal',
        'fontstretch' : 800
    }
    label_fonts = {
        'fontname' : 'sans-serif',
        'fontweight' : 500, 
        'fontsize' : 'large',
        'fontstyle' : 'normal',
        'fontvariant' : 'normal',
        'fontstretch' : 700
    }

    return title_fonts, label_fonts

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
    xticks_range = np.round(xticks_range, 2)

    return xticks_range, xmax, xmin

# Plot the correlations of the features with respect to a target column header in the dataset.
def barplot_features(df, x_label_desc='x label', remove_zeros=True, plot_size=(12, 10), sns_style='whitegrid', sns_context='talk', sns_palette='coolwarm', title=None):
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
    title : string, title description of the chart

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

    # Set the plot fonts
    title_fonts, label_fonts = set_fonts()

    # Create the plot
    plt.figure(figsize=plot_size)
    plt.xticks(xticks_range)
    plt.xlim(left=xmin, right=xmax)
    ax = sns.barplot(data=df_plot, x=df_plot.columns[0], y=df_plot.index.tolist(), palette=sns_palette)
    ax.set_xlabel(x_label_desc, **label_fonts)
    if title is not None:
        plt.title(title, **title_fonts)

# Plot the PCA features contributions chart with respect to a specified principal component index.
def barplot_features_pca(df_pca_comp, pc_index=1, x_label_desc='Variance contribution %', remove_zeros=True, plot_size=(12, 10), sns_style='whitegrid', sns_context='talk', sns_palette='coolwarm', title=None):
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
    title : string, title description of the chart

    Returns:
    -----------
    Display of bar chart
    """

    # Reshape input dataframe for plotting
    target_header = 'PC_' + str(pc_index)
    df = pd.DataFrame(data=df_pca_comp.iloc[pc_index - 1].sort_values(ascending=False))
    df.columns = [target_header]
    df[target_header] = df[target_header]**2
    df.sort_values(by=target_header, ascending=False)
    if df.shape[0] > 30:
        df_plot = df.head(30)
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

    # Set the plot fonts
    title_fonts, label_fonts = set_fonts()

    # Create the plot
    plt.figure(figsize=plot_size)
    plt.xticks(xticks_range)
    plt.xlim(left=xmin, right=xmax)
    ax = sns.barplot(data=df_plot, x=target_header, y=df_plot.index.tolist(), palette=sns_palette)
    ax.set_xlabel(x_label_desc + ' in principal component ' + str(pc_index), **label_fonts)
    if title is not None:
        plt.title(title, **title_fonts)

# Plot 2D scatter of PCA biplot
def scatter_pca(df_pca, target_header, pc_axes=(1, 2), sns_style='white', sns_context='talk', sns_palette='plasma', title='PCA scatter plot'):
    """
    Produce a PCA scatter plot.

    Arguments:
    -----------
    df_pca : pd.dataframe, PCA components dataframe as input data
    target_header : string, column header of the target label
    pc_axes : tuple, indicates the principal components to be assigned to the respective x and y axes
    sns_style : selection of builtin Seaborn set_style, background color theme categories (e.g. 'whitegrid', 'white', 'darkgrid', 'dark', etc)
    sns_context : selection of builtin Seaborn set_context, labels/lines categories (e.g. 'talk', 'paper', 'poster', etc)
    sns_palette : selection of builtin Seaborn palette, graph color theme categories (e.g. 'coolwarm', 'Blues', 'BuGn_r', etc, note adding '_r' at the end reverses the displayed color order)
    title : string, title description of the chart

    Returns:
    -----------
    Display of PCA scatter plot
    """
    
    # Define the style of the Seaborn plot
    sns.set_style(sns_style)
    sns.set_context(sns_context)

    # Set the plot fonts
    title_fonts, label_fonts = set_fonts()
    
    # Create the plot
    sns.lmplot(data=df_pca, x='PC_' + str(pc_axes[0]), y='PC_' + str(pc_axes[1]), hue=target_header, fit_reg=False, palette=sns_palette, size=8, aspect=1.5)
    ax = plt.gca()
    ax.set_title(title, **title_fonts)
    ax.set_xlabel('Principal component ' + str(pc_axes[0]), **label_fonts)
    ax.set_ylabel('Principal component ' + str(pc_axes[1]), **label_fonts)

# Plot 2D distribution normal
def distplot_features(df, feature_header, target_header_value=(None, None), bin_scale=0.5, plot_size=(10, 7), xlim=(None, None), theme_style='blue', title=None):
    """
    Produce a feature distributions plot against all target labels.

    Arguments:
    -----------
    df : pd.dataframe, input data for plotting
    feature_header : string, column header containing the feature labels to be plotted
    target_header_value : tuple, column header containing the target label, and the value of the target label
    bin_scale : float (between 0 and 1), the scaling factor to set the number of bins for the distribution plot
    plot_size : tuple, defines the size of the plot
    xlim : tuple, defines the x-axis limits for the plot
    theme : selection of 'red', 'blue', 'green', etc for colour themes of the plot
    title : string, title description of the chart

    Returns:
    -----------
    Display of distributions plot for the selected feature(s) in the dataset
    """
    
    # Set the plot size
    plt.figure(figsize=plot_size)
    
    # Set the plot colour themes
    theme = set_theme(theme_style)
    
    # Set the plot fonts
    title_fonts, label_fonts = set_fonts()

    # If the task is for producing a distribution plot of all range of feature values
    if target_header_value[0] is None or target_header_value[1] is None:
        plot_data = df[[feature_header]].dropna()
    # If the task is for producing a distribution plot of a selected range of feature values w.r.t. to a target description
    else:
        plot_data = df[df[target_header_value[0]] == target_header_value[1]][[feature_header]].dropna()

    plot_bins = int(round(plot_data[feature_header].max()*bin_scale))
    n_total = plot_data.shape[0]

    # Create the plot
    n, bins, patches = plt.hist(
        x=plot_data[feature_header], 
        bins=plot_bins, 
        color=theme['color'], 
        facecolor=theme['facecolor'], 
        edgecolor=theme['edgecolor'], 
        alpha=theme['alpha'], 
        linewidth=theme['linewidth'],
        linestyle=theme['linestyle'])
    
    if target_header_value[1] is not None:
        plt.legend(labels=[target_header_value[1]], loc=2, bbox_to_anchor=(1.05, 1), fontsize=label_fonts['fontsize'])
    else:
        plt.legend(labels=[feature_header], loc=2, bbox_to_anchor=(1.05, 1), fontsize=label_fonts['fontsize'])

    if xlim[0] is not None and xlim[1] is not None:
        n_samples = plot_data[(plot_data[feature_header] > xlim[0]) & (plot_data[feature_header] < xlim[1])].shape[0]
        plt.xlabel(feature_header + ' (samples displayed: ' + str(n_samples) + ', samples total: ' + str(n_total) + ')', **label_fonts)
        plt.xlim([xlim[0], xlim[1]])
    else:
        plt.xlabel(feature_header + ' (samples total: ' + str(n_total) + ')', **label_fonts)

    plt.ylabel('Frequency', **label_fonts)

    if title is not None:
        plt.title(title, **title_fonts)

# Plot 2D distribution kde
def kdeplot_features(df, feature_header, target_header=None, compare_labels=(None, None), plot_size=(10, 7), xlim=(None, None), sns_style='white', sns_context='talk', sns_palette='plasma', title=None):
    """
    Produce a feature distributions kde plot against all target labels.

    Arguments:
    -----------
    df : pd.dataframe, input data for plotting
    feature_header : string, column header containing the feature labels to be plotted
    target_header : string, column header of the target label
    compare_labels : tuple, target labels required for comparison in the plot
    plot_size : tuple, defines the size of the plot
    xlim : tuple, defines the x-axis limits for the plot
    sns_style : selection of builtin Seaborn set_style, background color theme categories (e.g. 'whitegrid', 'white', 'darkgrid', 'dark', etc)
    sns_context : selection of builtin Seaborn set_context, labels/lines categories (e.g. 'talk', 'paper', 'poster', etc)
    sns_palette : selection of builtin Seaborn palette, graph color theme categories (e.g. 'coolwarm', 'Blues', 'BuGn_r', etc, note adding '_r' at the end reverses the displayed color order)
    title : string, title description of the chart

    Returns:
    -----------
    Display of distributions kde plot for the selected feature(s) in the dataset
    """
    
    # Define the style of the Seaborn plot
    sns.set_style(sns_style)
    sns.set_context(sns_context)

    # Set the plot fonts
    title_fonts, label_fonts = set_fonts()

    # Create the plot
    plt.figure(figsize=plot_size)
    legend_labels = []
    plot_labels = None
    
    # If the task is for producing a vanilla distribution plot of a selected range of feature values
    if target_header is None:
        plot_data = df[feature_header].dropna()
        ax = sns.kdeplot(data=data_series)
        if xlim[0] is not None and xlim[1] is not None:
            plt.xlim([xlim[0], xlim[1]])

    # If the task is for producing a distribution plot of feature values against selected/all target value categories
    else:
        target_labels = df[target_header].unique()

        # If task requires comparing the distribution plots of feature values w.r.t. two different target value categories
        if compare_labels[0] is not None and compare_labels[1] is not None:
            plot_labels = compare_labels
            for label in plot_labels:
                plot_data = df.loc[df[target_header] == label][feature_header].dropna()
                ax = sns.distplot(a=plot_data, kde=True, kde_kws={'label' : label})
                legend_labels.append(label)
            handles, _ = ax.get_legend_handles_labels()
            ax.legend(handles, legend_labels, loc=2, bbox_to_anchor=(1.05, 1), fontsize=label_fonts['fontsize'])
            ax.set_xlabel(feature_header, **label_fonts)
            ax.set_ylabel('Frequency (normalised)', **label_fonts)
            if xlim[0] is not None and xlim[1] is not None:
                plt.xlim([xlim[0], xlim[1]])

        # If task is for distribution plots of feature values w.r.t. all target value categories.
        else:
            plot_labels = target_labels
            for label in plot_labels:
                plot_data = df.loc[df[target_header] == label][feature_header].dropna()
                ax = sns.kdeplot(data=plot_data)
                legend_labels.append(label)
            handles, _ = ax.get_legend_handles_labels()
            ax.legend(handles, legend_labels, loc=2, bbox_to_anchor=(1.05, 1), fontsize=label_fonts['fontsize'])
            ax.set_xlabel(feature_header, **label_fonts)
            ax.set_ylabel('Frequency (normalised)', **label_fonts)
            if xlim[0] is not None and xlim[1] is not None:
                plt.xlim([xlim[0], xlim[1]])

    if title is not None:
        plt.title(title, **title_fonts)
    
# Display PCA heatmap based on feature variance contribution across selected principal components
def heatmap_pca(df_pca_comp, n_features=3, n_comps=3, sns_cmap='plasma', annot=False, title=None):
    """
    Produce a PCA heatmap based on variance contributions of features across principal components using the Seaborn library.

    Arguments:
    -----------
    df_pca_comp : pd.dataframe, dataframe containing PCA components attribution as the input data
    n_features : integer, number of top features in variance contribution to include in the plot
    n_comps : integer, maximum number of principal components to be displayed in the heatmap
    sns_cmap : selection of 'hot', 'afmhot', 'gist_heat', 'viridis', 'plasma' etc, type of color map setting for heatmap
    annot : boolean, choice of true/false for display or not display value annotations on the heatmap
    title : string, title description of the chart

    Returns:
    -----------
    Display of heatmap
    """
    # Set header descriptions for displaying PCA results
    pc_headers = ['PC ' + str(pc_index + 1) for pc_index in range(n_comps)]

    # Preprocess pca components data
    features = []
    for pc_index in range(n_comps):
        comp_array = df_pca_comp.iloc[pc_index - 1].values**2
        # Get the top n features of each principal component
        feature_indices = np.argsort(-comp_array)[:n_features]
        new_features = []
        for feature_index in feature_indices:
            new_features.append(df_pca_comp.columns[feature_index])
            
        # Aggregate all top features across the selected principal components
        features = list(set(features).union(new_features))
    
    # Filter to get the relevant pca components based on the top features
    df_pca_comp = df_pca_comp.head(len(pc_headers)).set_index([pc_headers])
    df_pca_comp = df_pca_comp[features].transpose()
    df_pca_comp = df_pca_comp.applymap(np.square)*100
    
    # Plot the PCA heatmap
    # Set the plot fonts
    title_fonts, label_fonts = set_fonts()

    # Adjust the plot size w.r.t. features and principal components displayed
    if df_pca_comp.shape[1] <= 3:
        plot_width = 8
    elif df_pca_comp.shape[1] > 3 and df_pca_comp.shape[1] <= 6:
        plot_width = 10
    else:
        plot_width = 12
    
    if df_pca_comp.shape[0] <= 8:
        plot_height = df_pca_comp.shape[0]*0.9
    elif df_pca_comp.shape[0] > 8 and df_pca_comp.shape[0] <= 15:
        plot_height = df_pca_comp.shape[0]*0.6
    else:
        plot_height = df_pca_comp.shape[0]*0.4
    
    # Setup the plot
    plot_size = (plot_width, plot_height)
    plt.figure(figsize=plot_size)
    g = sns.heatmap(data=df_pca_comp, annot=annot, cmap=sns_cmap, cbar_kws={'label': 'Variance contribution %'})

    if title is not None:
        plt.title(title, **title_fonts)

    plt.xlabel(xlabel='Principal components', **label_fonts)
    plt.ylabel('Features', **label_fonts)
    
# Correlation matrix plot
def correlations_plot(df, plot_size=(10, 7), sns_style='white', sns_context='talk', sns_cmap='plasma', annot=False, title=None):
    """
    Produce a correlation matrix plot.

    Arguments:
    -----------
    df : pd.dataframe, dataframe as input data to be plotted
    plot_size : tuple, the setting of the plot size
    sns_style : selection of builtin Seaborn set_style, background color theme categories (e.g. 'whitegrid', 'white', 'darkgrid', 'dark', etc)
    sns_context : selection of builtin Seaborn set_context, labels/lines categories (e.g. 'talk', 'paper', 'poster', etc)
    sns_cmap : selection of 'hot', 'afmhot', 'gist_heat', 'viridis', 'plasma' etc, type of color map setting for heatmap
    annot : boolean, choice of true/false for display or not display value annotations on the heatmap
    title : string, title description of the chart

    Returns:
    -----------
    Display of correlation matrix plot
    """
    # calculate the correlation matrix
    corr = df.corr()

    # Set the plot fonts
    title_fonts, label_fonts = set_fonts()

    # Define the style of the Seaborn plot
    sns.set_style(sns_style)
    sns.set_context(sns_context)

    plt.figure(figsize=plot_size)

    # Plot the heatmap
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=annot, cmap=sns_cmap)

    if title is not None:
        plt.title(title, **title_fonts)
        
# Display lineplot
def lineplot_general(df, y_header_list, x_header=None, x_label_desc=None, y_label_desc=None, title=None, scaler=None, plot_size=(10, 7), sns_style='white', sns_context='talk', sns_palette=None):
    """
    Produce a general line plot.

    Arguments:
    -----------
    df : pd.dataframe, dataframe as input data to be plotted
    y_header_list : list, list of header descriptions of y series data
    x_header : string, header description of the x series data
    x_label_desc : string, label description on x-axis
    y_label_desc : string, label description on y-axis
    scaler : string, selection of 'standard', 'minmax' or 'robust', type of scaler used for data processing
    plot_size : tuple, the setting of the plot size
    sns_style : selection of builtin Seaborn set_style, background color theme categories (e.g. 'whitegrid', 'white', 'darkgrid', 'dark', etc)
    sns_context : selection of builtin Seaborn set_context, labels/lines categories (e.g. 'talk', 'paper', 'poster', etc)
    sns_palette : selection of builtin Seaborn palette, graph color theme categories (e.g. 'coolwarm', 'Blues', 'BuGn_r', etc, note adding '_r' at the end reverses the displayed color order)
    title : string, title description of the chart

    Returns:
    -----------
    Display of line plot
    """

    # Create appropriate input dataframe for Seaborn lineplot function
    df_plot = df[y_header_list].astype(np.float64)

    # Apply scaler to data
    if scaler is not None:
        if df_plot.shape[1] > 0:
            print('Scaling numerical data... ', end='')
            if scaler == 'standard':
                scaler_function = StandardScaler()
                scaler_function.fit(df_plot)
            elif scaler == 'minmax':
                scaler_function = MinMaxScaler()
                scaler_function.fit(df_plot)
            else:
                scaler_function = RobustScaler()
                scaler_function.fit(df_plot)
            df_plot_data = scaler_function.transform(df_plot)
            df_plot = pd.DataFrame(data=df_plot_data, columns=y_header_list)
            print('[Done]')
        
    # Create the plot
    plt.figure(figsize=plot_size)
    sns.set(context=sns_context)
    sns.set(style=sns_style)

    # Set the plot fonts
    title_fonts, label_fonts = set_fonts()

    # Set x series for lineplot function if applicable
    if x_header is not None:
        df_plot = pd.concat([df[[x_header]], df_plot], axis=1)
        for y_header in y_header_list:
            ax = sns.lineplot(x=x_header, y=y_header, data=df_plot, palette=sns_palette)
    else:
        ax = sns.lineplot(data=df_plot, palette=sns_palette)

    if title is not None:
        plt.title(title, **title_fonts)

    if x_label_desc is not None:
        plt.xlabel(xlabel=x_label_desc, **label_fonts)

    if y_label_desc is not None:
        if scaler is not None:
            y_label_desc = y_label_desc + ' (' + scaler + ' scaled)'
        plt.ylabel(ylabel=y_label_desc, **label_fonts)

    plt.legend(labels=y_header_list, loc=2, bbox_to_anchor=(1.05, 1), fontsize=label_fonts['fontsize'])
    
# Display barplot
def barplot_general(df, x_header, y_header, order='descending', xlabel_angle=45, plot_size=(10, 7), sns_style='white', sns_context='talk', sns_palette='coolwarm_r', title=None):
    """
    Produce a general barplot.

    Arguments:
    -----------
    df : pd.dataframe, dataframe as input data to be plotted
    x_header : string, column header of the x data (independent variable)
    y_header : string, column header of the y data (dependent variable)
    order : selection of 'descending', or 'ascending' for barplot display
    xlabel_angle : int, the degree of rotation of xlabel descriptions
    plot_size : tuple, the setting of the plot size
    sns_style : selection of builtin Seaborn set_style, background color theme categories (e.g. 'whitegrid', 'white', 'darkgrid', 'dark', etc)
    sns_context : selection of builtin Seaborn set_context, labels/lines categories (e.g. 'talk', 'paper', 'poster', etc)
    sns_palette : selection of builtin Seaborn palette, graph color theme categories (e.g. 'coolwarm', 'Blues', 'BuGn_r', etc, note adding '_r' at the end reverses the displayed color order)
    title : string, title description of the chart

    Returns:
    -----------
    Display of barplot
    """
    # Sort data prior to plotting
    if order == 'descending':
        ascending_order = False
    elif order == 'ascending':
        ascending_order = True
    else:
        ascending_order = False
        
    df = df.sort_values([y_header], ascending=ascending_order)
    plot_order = df[y_header].tolist()

    # Create the plot
    plt.figure(figsize=plot_size)
    sns.set(context=sns_context)
    sns.set(style=sns_style)

    # Set the plot fonts
    title_fonts, label_fonts = set_fonts()

    ax = sns.barplot(x=x_header, y=y_header, data=df, order=plot_order, palette=sns_palette)
    ax.set_xticklabels(df[target_header].tolist(), rotation=xlabel_angle)

    if title is not None:
        plt.title(title, **title_fonts)

    plt.xlabel(**label_fonts)
    plt.ylabel(**label_fonts)

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
def check_stats_scalers(df, headers):
    # Set scaled data headers
    original_header = header + '_original'
    std_scaled_header = header + '_standard_scaled'
    minmax_scaled_header = header + '_minmax_scaled'
    robust_scaled_header = header + '_robust_scaled'

    # Reshape the 1D input data into "transposed" column-wise array for use with sklearn scaler functions
    feature_series = df[header]
    feature_array = feature_series.values.reshape(-1, 1)

    # Fit data to scaler functions and get the scaled data after transformation
    std_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()
    robust_scaler = RobustScaler()
    std_scaled_array = std_scaler.fit_transform(feature_array)
    minmax_scaled_array = minmax_scaler.fit_transform(feature_array)
    robust_scaled_array = robust_scaler.fit_transform(feature_array)

    # Append the scaled data to a custom dataframe
    df_new = pd.DataFrame({original_header: feature_series})
    df_new[std_scaled_header] = std_scaled_array
    df_new[minmax_scaled_header] = minmax_scaled_array
    df_new[robust_scaled_header] = robust_scaled_array

    # Visualise original and scaled data distributions
    original_data = pd.Series(feature_series, name=original_header)
    std_scaled_data = pd.Series(
        df_new[std_scaled_header], name=std_scaled_header)
    minmax_scaled_data = pd.Series(
        df_new[minmax_scaled_header], name=minmax_scaled_header)
    robust_scaled_data = pd.Series(
        df_new[robust_scaled_header], name=robust_scaled_header)

    fig, ax = plt.subplots(2, 2, figsize=(15, 11))
    sns.kdeplot(original_data, ax=ax[0][0], shade=True, color='b')
    sns.kdeplot(std_scaled_data, ax=ax[0][1], shade=True, color='y')
    sns.kdeplot(minmax_scaled_data, ax=ax[1][0], shade=True, color='y')
    sns.kdeplot(robust_scaled_data, ax=ax[1][1], shade=True, color='y')
    return df_new

# Apply math operation scaling and output plots
def check_math_scalers(df, header):
    """
    Apply different scaler functions to a feature column using Sklearn python library.

    Arguments:
    -----------
    df : pd.dataframe, dataframe of the input data
    header : string, header description of the feature column subjected to the scaler functions

    Returns:
    -----------
    df_new : pd.dataframe, dataframe containing math function scaled values
    """
    # Set scaled data headers
    original_header = header + '_original'
    log_scaled_header = header + '_log_scaled'
    sqrt_scaled_header = header + '_sqrt_scaled'
    tanh_scaled_header = header + '_tanh_scaled'

    # Reshape the 1D input data into "transposed" column-wise array for use with sklearn scaler functions
    feature_series = df[header]

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
    df_new = pd.DataFrame({original_header: feature_series})
    df_new[log_scaled_header] = log_scaled_array
    df_new[sqrt_scaled_header] = sqrt_scaled_array
    df_new[tanh_scaled_header] = tanh_scaled_array

    # Visualise original and scaled data distributions
    original_data = pd.Series(feature_series, name=original_header)
    log_scaled_data = pd.Series(
        df_new[log_scaled_header], name=log_scaled_header)
    sqrt_scaled_data = pd.Series(
        df_new[sqrt_scaled_header], name=sqrt_scaled_header)
    tanh_scaled_data = pd.Series(
        df_new[tanh_scaled_header], name=tanh_scaled_header)

    fig, ax = plt.subplots(2, 2, figsize=(15, 11))
    sns.kdeplot(original_data, ax=ax[0][0], shade=True, color='b')
    sns.kdeplot(log_scaled_data, ax=ax[0][1], shade=True, color='y')
    sns.kdeplot(sqrt_scaled_data, ax=ax[1][0], shade=True, color='y')
    sns.kdeplot(tanh_scaled_data, ax=ax[1][1], shade=True, color='y')
    return df_new

# Filter the data to the range appropriate for outlier analysis
def outlier_filter(df, target_header, scale=1.5):
    """
    Helper function that outputs a filtered range of data for outlier consideration.
    
    Arguments:
    -----------
    df : pd.dataframe, original dataset
    target_header : string, the header description of columnn containing numeric values for outlier analysis
    scale : float, the factor multipled to the interquartile range for determining outlier range threshold

    Returns:
    -----------
    df_outlier_lower : pd.dataframe, resulting filtered lower outlier dataframe as output
    df_outlier_upper : pd.dataframe, resulting filtered upper outlier dataframe as output
    """
    q1 = df[target_header].quantile(0.25)
    q3 = df[target_header].quantile(0.75)
    iqr = q3 - q1
    threshold_lower = q1 - iqr*scale
    threshold_upper = q3 + iqr*scale

    df_outlier_lower = df[df[target_header] < threshold_lower]
    df_outlier_upper = df[df[target_header] > threshold_lower]
    
    return df_outlier_lower, df_outlier_upper

# Summary of the data characteristics for the filtered outlier data
def outlier_summary(df_outlier, df, feature_header, target_header, side, metric='mean', nsamples=10):
    """
    Helper function that outputs a summary table of outlier analysis.
    
    Arguments:
    -----------
    df_outlier : pd.dataframe, subset of data containing potential outlier samples
    df : pd.dataframe, original dataset
    feature_header : string, the header description of column containing features of which to perform outlier analysis
    target_header : string, the header description of columnn containing numeric values for outlier analysis
    side : selection of 'lower', 'upper', the side of extremity of the outlier analysis
    metric : selection of 'mean', 'freq', the metric to sort in descending order with in the outlier analysis summary table
    nsamples : int, number of top features to show in the outlier analysis summary table

    Returns:
    -----------
    df_summary : pd.dataframe, resulting dataframe as output
    """
    print('Initializing outlier summary table... ', end='')
    df_summary = unique_values(df_outlier, feature_header).head(100)
    df_summary.rename(columns={df_summary.columns[0] : feature_header}, inplace=True)
    df_summary.rename(columns={'Relative representation %' : 'Freq % (outlier)'}, inplace=True)
    norm_freq_frac = []
    outlier_means = []
    norm_means = []
    outlier_stddevs = []
    norm_stddevs = []
    norm_mean = df[target_header].mean()
    norm_stddev = df[target_header].std()
    summary_rows = df_summary.shape[0]
    print('[Done]')
    
    print('\rUpdating outlier table with feature metrics... progress: 0% ... ', end='')
    i = 1
    for label in df_summary.iloc[:, 0]:
        percent_progress = round(100*(i/summary_rows))
        if percent_progress % 5 == 0:
            print('\rUpdating outlier table with feature metrics... progress: ' + str(percent_progress) + '% ... ', end='')

        freq_frac = 100*df[df[feature_header]==label][feature_header].shape[0]/df.shape[0]
        outlier_mean = df[df[feature_header]==label][target_header].mean()
        outlier_stddev = df[df[feature_header]==label][target_header].std()
        norm_freq_frac.append(freq_frac)
        outlier_means.append(outlier_mean)
        outlier_stddevs.append(outlier_stddev)
        norm_means.append(norm_mean)
        norm_stddevs.append(norm_stddev)
        i += 1
    
    df_summary['Freq %'] = norm_freq_frac
    df_summary['Freq ratio'] = df_summary['Freq % (outlier)']/df_summary['Freq %']
    df_summary['Mean (outlier)'] = outlier_means
    df_summary['Std dev (outlier)'] = outlier_stddevs
    df_summary['Mean'] = norm_means
    df_summary['Std dev'] = norm_stddevs
    print('[Done]')

    if metric=='mean':
        if side=='lower':
            df_summary = df_summary.sort_values(by='Mean (outlier)', ascending=True).head(nsamples)
        elif side=='upper':
            df_summary = df_summary.sort_values(by='Mean (outlier)', ascending=False).head(nsamples)
    elif metric=='freq':
        df_summary = df_summary.sort_values(by='Freq ratio', ascending=False).head(nsamples)
    else:
        if side=='lower':
            df_summary = df_summary.sort_values(by='Mean (outlier)', ascending=True).head(nsamples)
        elif side=='upper':
            df_summary = df_summary.sort_values(by='Mean (outlier)', ascending=False).head(nsamples)

    return df_summary

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
