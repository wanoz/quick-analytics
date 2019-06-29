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
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, kurtosis, skew
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Imputer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, KFold, ShuffleSplit
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, roc_curve, precision_recall_curve, roc_auc_score, auc, average_precision_score
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
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
def scan_dir(file_dir, file_list=[]):
    # Recursively search through the directory tree and append file name and file path 
    try:
        # For native directory files
        for name in os.listdir(file_dir):
            path = os.path.join(file_dir, name)
            if os.path.isfile(path):
                file_list.append((name, path))
            else:
                scan_dir(path, file_list)
    except:
        try:
            # For Google Drive files
            file_dir = os.path.join(file_dir, 'drive')
            for name in files.os.listdir(file_dir):
                path = os.path.join(file_dir, name)
                if os.path.isfile(path):
                    file_list.append((name, path))
                else:
                    scan_dir(path, file_list)
        except:
            pass

    return file_list

# Secondary helper function for file search
def locate_file(file_name, file_type):
    base_dir = os.path.dirname(os.path.realpath('__file__'))
    
    file_name_base = file_name.lower()
    original_file_name = None
    file_dir = None

    # Get a list containing all the files within the base directory tree
    file_list = scan_dir(base_dir)

    # Look for the specific file in the file list that matches the required file name
    for f_name, f_dir in file_list:
        if file_type == 'csv_excel':
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
        elif file_type == 'shape':
            if f_name.endswith('.shp'):
                f_name_base = f_name.replace('.shp', '').lower()
                if file_name_base == f_name_base:
                    original_file_name = f_name
                    file_dir = f_dir
                    break

    return original_file_name, file_dir

# Read raw data into Pandas dataframe for analysis.
def read_data(file_name, file_type='csv_excel', encoding='utf-8', sheet_name=0):
    """
    Read data into dataframe for analysis.

    Arguments:
    -----------
    file_name : string, name of the data file (excludes extension descriptions e.g. '.csv', '.xlsx' or 'xls')
    file_type : selectio of "csv_excel" or "shape" for reading either csv/excel file or geolocation shape file
    sheet_name : int or string, the Excel sheet index or the sheet name containing the data to be read

    Returns:
    -----------
    df_read : pd.dataframe, the dataframe read from the dataset
    """
    # Get the directory of the data file
    original_file_name, file_dir = locate_file(file_name, file_type)

    # Read the file content into a dataframe
    df_read = None
    if file_dir is not None:
        if file_type == 'csv_excel':
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
                    if sheet_name != 0:
                        print('Status: "' + original_file_name + '", worksheet: "' + sheet_name + '" has been successfully read into dataframe!')
                    else:
                        xl = pd.ExcelFile(file_dir)
                        print('Status: "' + original_file_name + '", worksheet: "' + str(xl.sheet_names[0]) + '" has been successfully read into dataframe!')
                except:
                    print('Status: "' + original_file_name + '" cannot be read into dataframe. Note: Excel file format is detected (ensure sheetname of the content is titled "sheet1".')
                    raise
            elif file_dir.endswith('.xls'):
                try:
                    df_read = pd.read_excel(file_dir, sheet_name=sheet_name)
                    if sheet_name != 0:
                        print('Status: "' + original_file_name + '", worksheet: "' + sheet_name + '" has been successfully read into dataframe!')
                    else:
                        xl = pd.ExcelFile(file_dir)
                        print('Status: "' + original_file_name + '", worksheet: "' + str(xl.sheet_names[0]) + '" has been successfully read into dataframe!')
                except:
                    print('Status: "' + original_file_name + '" cannot be read into dataframe. Note: Excel file format is detected (ensure sheetname of the content is titled "sheet1".')
                    raise
        elif file_type == 'shape':
            if file_dir.endswith('.shp'):
                try:
                    df_read = gpd.read_file(file_dir)
                    print('Status: "' + original_file_name + '" has been successfully read!')
                except:
                    print('Status: "' + original_file_name + '" cannot be read. Note: Ensure the shapefile is correctly setup.')
                    raise
        else:
            print('Status: The specified file type parameter is not recognized.')
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

# Helper function for labelling duplicate count index information
def label_duplicates(df, feature_header, duplicate_position='last', order='ascending'):
    '''
    Helper function that produces an updated dataset containing the count index of duplicated values in the selected column.

    Arguments:
    -----------
    df : pd.dataframe, dataframe to be passed as input
    feature_header : string, the column with the header description that will be inspected for the count index information of duplicated values
    duplicate_position : selection of 'last', 'first', the additional information of first or last description tagged to the count index
    order : string, selection of 'ascending', 'descending', the order of which the dataframe will be arranged prior to duplicates labelling
    
    Returns:
    -----------
    df_output : pd.dataframe, resulting dataframe as output

    '''
    
    if order == 'ascending':
        df = df.sort_values([feature_header])
        print('Sorted dataset based on feature column in ascending order')
    elif order == 'descending':
        df = df.sort_values([feature_header], ascending=False)
        print('Sorted dataset based on feature column in descending order')
    elif order == 'original':
        pass
    df = df.reset_index(drop=True)

    print('Labelling duplicate data... ', end='')
    value_lookup = {}
    duplicate_index_list = []
    duplicate_position_list = []
    duplicate_label_window = []
    df_output = df
    
    current_id = ''
    prev_id = ''
    unique_count = 1
    max_row = df.shape[0]
    row_count = 1
    
    # Iterate through the data of the selected column to apply label on duplicates
    for index, row in df.iterrows():
        current_id = str(row[feature_header])
        
        if duplicate_position == 'first':
            if prev_id == '':
                duplicate_index_list.append(unique_count)
                duplicate_position_list.append('first')
                unique_count += 1
            else:
                if current_id != prev_id:
                    unique_count = 1
                    duplicate_index_list.append(unique_count)
                    duplicate_position_list.append('first')
                else:
                    unique_count += 1
                    duplicate_index_list.append(unique_count)
                    duplicate_position_list.append(np.nan)
                    
        elif duplicate_position == 'last':
            if prev_id == '':
                duplicate_index_list.append(unique_count)
                duplicate_position_list.append(np.nan)
                unique_count += 1
            else:
                if current_id != prev_id:
                    unique_count = 1
                    duplicate_index_list.append(unique_count)
                    duplicate_position_list[len(duplicate_position_list) - 1] = 'last'
                    duplicate_position_list.append(np.nan)
                else:
                    unique_count += 1
                    duplicate_index_list.append(unique_count)
                    duplicate_position_list.append(np.nan)
            if row_count == max_row:
                duplicate_position_list[len(duplicate_position_list) - 1] = 'last'
            
        else:
            print('\nDuplicate position input argument is not recognized - please specify either "first" or "last"')
            break
                    
        prev_id = current_id
        row_count += 1
            
    # Add the duplicate index, duplicate position labels in the output dataset
    df_output[feature_header + ' (duplicate #)'] = pd.Series(duplicate_index_list)
    df_output[feature_header + ' (duplicate position)'] = pd.Series(duplicate_position_list)

    print('[Done]')

    return df_output

# Helper function to extract time relevant data features by grouping of specific identifier/label
def group_time_features(df, time_header, group_header, time_measure='day'):
    """
    Helper function that extracts useful time related data features by grouping of specified identifier/label

    Arguments:
    -----------
    df : pd.dataframe, dataframe to be passed as input
    time_header : string, the column header description containing the time measurement data
    group_header : string, the column header description of the grouping label.
    time_measure : string, selection of 'day', 'hour', 'month' or 'year', to specify the type of time measurement for the data feature value

    Returns:
    -----------
    df_output : pd.dataframe, resulting dataframe as output
    """
    current_name = ''
    prev_name = ''
    current_date = ''
    prev_date = ''
    time_prev = []
    time_prev_mavg = []
    time_prev_msum = []
    time_prev_mmax = []
    time_prev_mmin = []

    # Extract time measure values based on grouping label
    for index, row in df.iterrows():
        current_date = row[time_header]
        current_name = str(row[group_header])
        if current_name == '':
            time_prev.append(0)
            time_prev_mavg.append(0)
            time_prev_msum.append(0)
            time_prev_mmax.append(0)
            time_prev_mmin.append(0)
            time_prev_window = []
        else:
            if current_name != prev_name:
                time_prev.append(0)
                time_prev_mavg.append(0)
                time_prev_msum.append(0)
                time_prev_mmax.append(0)
                time_prev_mmin.append(0)
                time_prev_window = []
            else:
                if time_measure == 'day':
                    time_diff = (prev_date - current_date).days
                    time_prev_window.append(time_diff)
                    
                    time_prev.append(time_diff)
                    time_prev_mavg.append(np.mean(time_prev_window))
                    time_prev_msum.append(np.sum(time_prev_window))
                    time_prev_mmax.append(np.max(time_prev_window))
                    time_prev_mmin.append(np.min(time_prev_window))
                elif time_measure == 'hour':
                    time_diff = (prev_date - current_date).days*24
                    time_prev_window.append(time_diff)
                    
                    time_prev.append(time_diff)
                    time_prev_mavg.append(np.mean(time_prev_window))
                    time_prev_msum.append(np.sum(time_prev_window))
                    time_prev_mmax.append(np.max(time_prev_window))
                    time_prev_mmin.append(np.min(time_prev_window))
                elif time_measure == 'month':
                    time_diff = (prev_date - current_date).days/30.4375
                    time_prev_window.append(time_diff)
                    
                    time_prev.append(time_diff)
                    time_prev_mavg.append(np.mean(time_prev_window))
                    time_prev_msum.append(np.sum(time_prev_window))
                    time_prev_mmax.append(np.max(time_prev_window))
                    time_prev_mmin.append(np.min(time_prev_window))
                elif time_measure == 'year':
                    time_diff = (prev_date - current_date).days/365.25
                    time_prev_window.append(time_diff)
                    
                    time_prev.append(time_diff)
                    time_prev_mavg.append(np.mean(time_prev_window))
                    time_prev_msum.append(np.sum(time_prev_window))
                    time_prev_mmax.append(np.max(time_prev_window))
                    time_prev_mmin.append(np.min(time_prev_window))
                else:
                    print('Time measure input is not recognized - please specify either "day", "hour", "month" or "year"')
                    break

        prev_name = current_name
        prev_date = current_date

    # Form final output dataframe
    df_features = pd.DataFrame({str(time_measure) + ' since previous record' : time_prev, 
                                str(time_measure) + ' since previous record (mavg)' : time_prev_mavg, 
                                str(time_measure) + ' since previous record (msum)' : time_prev_msum, 
                                str(time_measure) + ' since previous record (mmax)' : time_prev_mmax, 
                                str(time_measure) + ' since previous record (mmin)' : time_prev_mmin})
    df_output = pd.concat([df, df_features], axis=1)

    return df_output

# Transform data containing sequence based data values into structured relational format
def sequence_encoding(df, id_header, feature_headers, id_repeats_max_limit=10):
    '''
    Helper function that transforms sequence based data values into structured relational format.

    Arguments:
    -----------
    df : pd.dataframe, dataframe to be passed as input
    id_header : str, the header description of the column containing ID group of the sequence data
    feature_headers : list of str, list of header descriptions of the columns containing the sequence data to be extracted
    id_repeats_max_limit : int, the maximum number of sequence orders to be extracted

    Returns:
    -----------
    df_output : pd.dataframe, resulting dataframe as output
    
    '''
    id_repeats = 0
    current_id = None
    prev_id = None
    output_series = {}
    
    id_count = 0
    row_count = 0
    row_count_max = df.shape[0]
    percent_progress = 0
    
    for index, row in df.iterrows():
        if row_count == 0:
            current_id = row[id_header]
        
            # Initialise data on multiple series sequences (at first row)
            output_series[id_header] = []
            output_series[id_header].append(row[id_header])
            
            for header in feature_headers:
                for id_index in range(id_repeats_max_limit):
                    output_series[header + '_seq' + str(id_index)] = [np.nan]
                    
            # Update data on multiple series sequences
            for header in feature_headers:
                for id_index in range(id_repeats_max_limit):
                    if id_index == id_repeats:
                        output_series[header + '_seq' + str(id_index)][id_count - 1] = row[header]
                    else:
                        output_series[header + '_seq' + str(id_index)][id_count - 1] = np.nan
            
            id_count += 1
            prev_id = current_id
        else:
            # Check for next id that is repeating
            current_id = row[id_header]
            
            if current_id == prev_id:
                id_repeats += 1
                # Update data on multiple series sequences
                for header in feature_headers:
                    for id_index in range(id_repeats_max_limit):
                        if id_index == id_repeats:
                            output_series[header + '_seq' + str(id_index)][id_count - 1] = row[header]
            else:
                id_repeats = 0
                id_count += 1
                # Append data on multiple series sequences
                output_series[id_header].append(row[id_header])
                
                for header in feature_headers:
                    for id_index in range(id_repeats_max_limit):
                        if id_index == id_repeats:
                            output_series[header + '_seq' + str(id_index)].append(row[header])
                        else:
                            output_series[header + '_seq' + str(id_index)].append(np.nan)
            
            prev_id = current_id
            
        row_count = row_count + 1
        percent_progress = round((float(row_count)/float(row_count_max))*100, 2)
        if percent_progress % 1 == 0:
            print('\rData feature extraction in progress: ' + str(int(percent_progress)) + '%', end='')
            
    df_output = pd.DataFrame(output_series)
    
    return df_output

# Downsample data containing drastically higher number of negative label values compared to positive values (for model training)
def downsample(df, target_header, frac=0.3, weight_scaling=0):
    '''
    Helper function that downsamples a dataset and apply an weighted scalar to numerical values based on the downsampling ratio.

    Arguments:
    -----------
    df : pd.dataframe, dataframe to be passed as input
    target_header : string, the header description of the column containing the target label
    frac : float, fraction of downsampling desired, e.g. downsampled data that is 10% of the original dataset size is expressed as 0.1
    weight_scaling : float, {0, 1} magnitude of additional numerical weights to data after downsampling for dataset calibration.

    Returns:
    -----------
    df_output : pd.dataframe, resulting dataframe as output
    
    '''
    
    df_negative = df[df[target_header] == 0]
    df_positive = df[df[target_header] == 1]
    
    neg_samples = df_negative.shape[0]
    pos_samples = df_positive.shape[0]

    print('Negative label has ' + str(neg_samples) + ' samples.')
    print('Positive label has ' + str(pos_samples) + ' samples.')

    exit_function = False

    if neg_samples > pos_samples:
        n_samples = round(df_negative.shape[0]*frac)
        print('Extracted ' + str(n_samples) + ' record samples from negative label data.')
    elif neg_samples < pos_samples:
        n_samples = round(df_positive.shape[0]*frac)
        print('Extracted ' + str(n_samples) + ' record samples from positive label data.')
    else:
        print('Negative and positive label samples are of equal size. Downsampling in the context of this function is only appropriate for dataset with imbalanced class distribution.')
        exit_function = True
    
    if exit_function is False:
        # Downsample dataset
        df_negative_samples = df_negative.sample(n=n_samples)
        
        # Apply weighting to downsampled dataset
        if weight_scaling > 1:
            weight_scaling = 1
        elif weight_scaling < 0:
            weight_scaling = 0

        df_negative_weighted = df_negative_samples.select_dtypes(include=[np.number])*int(round((1/frac)**weight_scaling))
        df_negative_samples[df_negative_weighted.columns] = df_negative_weighted
        
        df_output = pd.concat([df_negative_samples, df_positive])
        df_output.reset_index(drop=True, inplace=True)
    
    return df_output

# Calculate values of rolling window outputs for time series analysis
def rolling_window_features(df, feature_headers, window_size=10, window_functions=['mean']):
    '''
    Helper function to process values in a rolling window fashion within a time series dataset

    Arguments:
    -----------
    df : pd.dataframe, dataframe to be passed as input
    feature_headers : list, the descriptions of column headers that contains numerical value to be processed within the rolling window regime
    window_size : int, the specified size of the rolling window
    window_functions : list of 'max', 'min', 'sum', 'mean', the set of calculation functions to be applied within the rolling window regime

    Returns:
    -----------
    df_output : pd.dataframe, resulting dataframe as output
    '''
    feature_window = {}
    feature_window_ref = {}
    feature_output = {}
    sum_window = {}
    mean_window = {}
    row_count = 0
    
    print('Processing rolling window period: ' + str(window_size))
    print('Processing rolling window calculations: ' + str(window_functions))
    
    # Initialise output entries
    for header in feature_headers:
        for func in window_functions:
            feature_window[header + ' rolling ' + str(window_size) + ' sample ' + func] = []
            feature_output[header + ' rolling ' + str(window_size) + ' sample ' + func] = []
            
        feature_window_ref[header] = []
        sum_window[header] = 0
        mean_window[header] = 0
    
    # Apply rolling window processing through data
    for index, row in df.iterrows():
        row_count += 1
        for header in feature_headers:
            if row_count <= window_size:
                for func in window_functions:
                    if func == 'max':
                        feature_output[header + ' rolling ' + str(window_size) + ' sample ' + func].append(0)

                    elif func == 'min':
                        feature_output[header + ' rolling ' + str(window_size) + ' sample ' + func].append(0)
                        
                    elif func == 'sum':
                        feature_output[header + ' rolling ' + str(window_size) + ' sample ' + func].append(0)
                        sum_window[header] = sum_window[header] + row[header]
                        
                    elif func == 'mean':
                        feature_output[header + ' rolling ' + str(window_size) + ' sample ' + func].append(0)
                        mean_window[header] = (mean_window[header]*(row_count - 1) + row[header])/row_count
                        
                feature_window_ref[header].append(row[header])
            else:
                for func in window_functions:
                    feature_window[header + ' rolling ' + str(window_size) + ' sample ' + func] = feature_window_ref[header].copy()
                    if func == 'max': 
                        feature_window[header + ' rolling ' + str(window_size) + ' sample ' + func].sort(reverse=True)
                        feature_output[header + ' rolling ' + str(window_size) + ' sample ' + func].append(feature_window[header + ' rolling ' + str(window_size) + ' sample ' + func][0])

                    if func == 'min':
                        feature_window[header + ' rolling ' + str(window_size) + ' sample ' + func].sort(reverse=False) 
                        feature_output[header + ' rolling ' + str(window_size) + ' sample ' + func].append(feature_window[header + ' rolling ' + str(window_size) + ' sample ' + func][0])
                        
                    if func == 'sum':
                        sum_window[header] = sum_window[header] + row[header] - feature_window_ref[header][0]
                        feature_output[header + ' rolling ' + str(window_size) + ' sample ' + func].append(sum_window[header])
                    
                    if func == 'mean':
                        mean_window[header] = (mean_window[header]*window_period + row[header] - feature_window_ref[header][0])/window_size
                        feature_output[header + ' rolling ' + str(window_size) + ' sample ' + func].append(mean_window[header])
                        
                feature_window_ref[header].append(row[header])
                feature_window_ref[header] = feature_window_ref[header][1:]

    # Finalise output data
    df_feature_output = pd.DataFrame(feature_output)
    df_output = pd.concat([df, df_feature_output], axis=1)
    
    return df_output

# Helper function to apply custom log transform - (-1)*log(abs(x) + 1) if negative or (log(abs(x) + 1) if positive
def transform_log(df, target_header):
    """
    Apply log transform to data to facilitate modelling.

    Arguments:
    -----------
    df : pd.dataframe, dataframe to be passed as input
    target_header : string, the header description of the column containing the target label

    Returns:
    -----------
    df_output : pd.dataframe, resulting dataframe as output
    """
    
    # Separate features and target data
    df_x = df.drop(columns=[target_header])
    df_y = df[[target_header]]
    
    # Isolate sub-dataset containing categorical values
    categorical = df_x.loc[:, df_x.dtypes == object].copy()
    
    # Isolate sub-dataset containing non-categorical values
    non_categorical = df_x.loc[:, df_x.dtypes != object].copy()
    non_categorical = non_categorical.astype(np.float64)
    
    feature_headers = non_categorical.columns.tolist()
    
    # Store transformed data into dictionary
    transformed_data = {}
    
    # Apply log transformation throughout the data
    for header in feature_headers:
        transformed_values = []
        for index, row in non_categorical.iterrows():
            log_val = 0
            if row[header] < 0:
                log_val = (-1)*np.log(np.abs(row[header]) + 1)
            else:
                log_val = np.log(np.abs(row[header]) + 1)
            transformed_values.append(log_val)
        transformed_data[header] = transformed_values
    
    non_categorical = pd.DataFrame(transformed_data)
    
    # Finalise dataset
    df_x = pd.concat([non_categorical, categorical], axis=1)
    df_output = pd.concat([df_x, df_y], axis=1)
    
    return df_output

# Transform data via estimated fit to a polynomial function
def transform_polyfit(df, degree=3, output_length=None):
    '''
    Helper function that transforms data values via estimated fit to a polynomial function.

    Arguments:
    -----------
    df : pd.dataframe, dataframe to be passed as input
    degree : int, the degree of the polynomial specified for the fitting function
    output_length : int, the length of the x values for the transformed output dataset

    Returns:
    -----------
    df_output : pd.dataframe, resulting dataframe as output
    
    '''
    
    # Get x value intervals required for polyfit function
    x_fit = np.linspace(1, df.shape[0], num=df.shape[0])
    
    # Set x value output intervals desired
    x_output = x_fit
    if output_length is not None:
        x_output = np.linspace(1, df.shape[0], num=output_length)
        
    # Initialise output dataframe
    df_output = pd.DataFrame({'Test column' : [i for i in range(len(x_output))]})
        
    # Get column headers
    headers = df.columns.tolist()
    
    # Iterate through each column header to fit data series to polynomial function
    for header in headers:
        y_fit = df[header]
        coeff = np.polyfit(x_fit, y_fit, degree)
        function = np.poly1d(coeff)
        y_est = function(x_output)
        df_temp = pd.DataFrame({header : y_est})
        df_output = pd.concat([df_output, df_temp], axis=1)
    
    df_output.drop(columns=['Test column'], inplace=True)
    
    return df_output

# Secondary helper function for categorical value encoding, numerical imputation and scaling
def transform_data(df, target_header, numerical_imputer=None, scaler=None, encoder=None, remove_binary=None):
    # Separate features and target data
    df_y = df[[target_header]].copy()
    df_x = df.drop(columns=[target_header]).copy()
    
    # Isolate sub-dataset containing categorical values
    categorical = df_x.loc[:, df_x.dtypes == object].copy()
    
    # Isolate sub-dataset containing non-categorical values
    non_categorical = df_x.loc[:, df_x.dtypes != object].copy()
    non_categorical = non_categorical.astype(np.float64)
    
    if remove_binary:
        print('Inspecting data values... ', end='')
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
        non_categorical.reset_index(drop=True, inplace=True)
        categorical.reset_index(drop=True, inplace=True)
        df_x = pd.concat([non_categorical, categorical], axis=1)
    elif non_categorical.shape[1] == 0:
        df_x = categorical
    else:
        df_x = non_categorical

    return df_x, df_y

# Helper function for filter column headings with inclusion/exclusion of keywords
def filter_column_headers(df, include_words=None, exclude_words=None, target_header=None):
    """
    Helper function for filter column headings with inclusion/exclusion of keywords in the dataset

    Arguments:
    -----------
    df : pd.dataframe, dataframe to be passed as input
    include_words : list, list of keyword strings to include in the filter
    exclude_words : list, list of keyword strings to exclude in the filter
    target_header : string, the header description of the column containing the prediction target label

    Returns:
    -----------
    df_output : pd.dataframe, resulting dataframe as output
    """
    
    column_list = df.columns.tolist()
    filtered_column_list = []
    
    # Create filtered column headers based on include keywords list
    for column_header in column_list:
        if include_words is not None and len(include_words) > 0:
            for include_word in include_words:
                if include_word.lower() in column_header.lower():
                    if column_header not in filtered_column_list:
                        filtered_column_list.append(column_header)
        else:
            filtered_column_list = column_list

    # Remove additional words based on exclude keywords list
    remove_list = []
    for column_header in filtered_column_list:
        if exclude_words is not None and len(exclude_words) > 0:
            for exclude_word in exclude_words:
                if exclude_word.lower() in column_header.lower():
                    remove_list.append(column_header)
                    
    # Prepare column list
    filtered_column_list = list(set(filtered_column_list) - set(remove_list))
    if (target_header is not None) and (target_header not in filtered_column_list):
        filtered_column_list = filtered_column_list + [target_header]

    # Apply column filter
    df_output = df[filtered_column_list]
    
    return df_output

# Helper function for filter column headings with inclusion/exclusion of keywords
def filter_quantile(df, target_header, quantile_range=(0.25, 0.75)):
    """
    Helper function for filter column headings with inclusion/exclusion of keywords in the dataset

    Arguments:
    -----------
    df : pd.dataframe, dataframe to be passed as input
    target_header : integer, the column containing the values for filtering
    quantile_range : tuple, the lower and upper limit values for specifying the filter range

    Returns:
    -----------
    df_output : pd.dataframe, resulting dataframe as output
    """
    
    lower_val = df[target_header].quantile(quantile_range[0])
    upper_val = df[target_header].quantile(quantile_range[1])

    # Apply the filter
    df_output = df[(df[target_header] >= lower_val) & (df[target_header] <= upper_val)]
    df_output = df_output.reset_index(drop=True)
    
    return df_output

# Obtain normality, t-test scores on independent variables with respect to binary class groups
def feature_class_stats(df, target_header, p_value_threshold=0.05):
    """
    Helper function for determining feature statistical significance between negative and positive binary class groups.

    Arguments:
    -----------
    df : pd.dataframe, dataframe to be passed as input
    target_header : string, the header description of the column containing the target label
    p_value_threshold : float, the specified p-value for null hypothesis test (below the threshold means rejection of null hypothesis)

    Returns:
    -----------
    df_output : pd.dataframe, resulting dataframe as output
    """
    
    df_positive = df[df[target_header] == 1]
    df_negative = df[df[target_header] == 0]
    
    # Separate features and target data
    df_positive_x = df_positive.drop(columns=[target_header])
    df_negative_x = df_negative.drop(columns=[target_header])
    
    # Isolate sub-dataset containing categorical values
    categorical_positive = df_positive_x.loc[:, df_positive_x.dtypes == object].copy()
    categorical_negative = df_negative_x.loc[:, df_negative_x.dtypes == object].copy()
    
    # Isolate sub-dataset containing non-categorical values
    non_categorical_positive = df_positive_x.loc[:, df_positive_x.dtypes != object].copy()
    non_categorical_positive = non_categorical_positive.astype(np.float64)
    non_categorical_negative = df_negative_x.loc[:, df_negative_x.dtypes != object].copy()
    non_categorical_negative = non_categorical_negative.astype(np.float64)
    
    feature_headers = non_categorical_positive.columns.tolist()
    n_samples = df.shape[0]
    skewness_max = 2*(6/n_samples)**(1/2)
    kurtosis_max = 2*(24/n_samples)**(1/2)
    
    # Get feature statistics with respect to binary class 
    feature_desc = []
    feature_mean_positive = []
    feature_mean_negative = []
    feature_mean_diff = []
    t_scores = []
    p_values = []
    stat_significance = []
    data_normality = []
    for header in feature_headers:
        samples_positive = non_categorical_positive[header]
        samples_negative = non_categorical_negative[header]
        t_score, p_value = ttest_ind(samples_positive, samples_negative)
        skewness_positive = np.abs(skew(samples_positive))
        skewness_negative = np.abs(skew(samples_negative))
        kurtosis_positive = np.abs(kurtosis(samples_positive))
        kurtosis_negative = np.abs(kurtosis(samples_negative))
        
        normality_flag = False
        # Assess normality
        if (skewness_positive > skewness_max) or (skewness_negative > skewness_max) or \
            (kurtosis_positive > kurtosis_max) or (kurtosis_negative > kurtosis_max):
            normality_flag = True
            
        # Assess statistical significance
        stat_significance_flag = False
        if (p_value < p_value_threshold) and (normality_flag == True):
            stat_significance_flag = True
        
        # Gather stats scores
        feature_desc.append(header)
        feature_mean_positive.append(samples_positive.mean())
        feature_mean_negative.append(samples_negative.mean())
        feature_mean_diff.append(100*round((samples_positive.mean() - samples_negative.mean())/samples_negative.mean(), 4))
        t_scores.append(t_score)
        p_values.append(p_value)
        stat_significance.append(stat_significance_flag)
        data_normality.append(normality_flag)
        
    # Finalise output
    df_output = pd.DataFrame({'Feature description' : feature_desc, 
                              'Neg class mean' : feature_mean_negative, 
                              'Pos class mean' : feature_mean_positive, 
                              'Diff in mean (%)' : feature_mean_diff,
                              'T-score (Neg/Pos)' : t_scores, 
                              'P-value (Neg/Pos)' : p_values, 
                              'Stat significance' : stat_significance,
                              'Data normality (Neg/Pos)' : data_normality})
    
    df_output = df_output.sort_values(by='P-value (Neg/Pos)', ascending=True)
    df_output = df_output.sort_values(by=['Data normality (Neg/Pos)', 'Stat significance'], ascending=False)
    df_output.reset_index(drop=True, inplace=True)
    print('Statistical significance threshold for P-value: ' + str(p_value_threshold))
    
    return df_output
    
# Helper function for centroids distance association calculation
def centroids_features(df, target_header, target_cluster_label=1, metric='euclidean', shrink_threshold=None, numerical_imputer=None, scaler=None, encoder=None, remove_binary=None):
    """
    Helper function for centroids distance association calculation in the dataset

    Arguments:
    -----------
    df : pd.dataframe, dataframe to be passed as input
    target_header : int, the column containing the cluster label in integer format
    metric : string, selection of 'euclidean' (minimizing sum of distances), 'manhattan' (median of distances)
    shrink_threshold : float, threshold for shrinking centroids to remove features
    numerical_imputer : selection of 'mean', 'median', 'most_frequent', the type of imputation strategy for processing missing data
    scaler : string, selection of 'standard', 'minmax' or 'robust', type of scaler used for data processing
    encoder : selection of 'one_hot', 'label_encoding', the type of encoding method for categorical data
    remove_binary : boolean, option to remove columns containing binary values

    Returns:
    -----------
    df_output : pd.dataframe, resulting dataframe as output
    """

    # Apply the optional data transformation (imputing, scaling, encoding) if required 
    df_x, df_y = transform_data(df, target_header, numerical_imputer, scaler, encoder, remove_binary)

    print('Calculating data centroids... ', end='')
    # Prepare data
    y_train = df_y.values
    X_train = df_x.values

    df_output = df
    target_centroid = None

    # Fit data to model
    model = NearestCentroid(metric=metric, shrink_threshold=shrink_threshold)
    model.fit(X_train, y_train.ravel())
    
    centroids = model.centroids_
    centroid_labels = [model.predict([centroid])[0] for centroid in centroids]

    df_centroids = pd.DataFrame({'Centroid' : centroids.tolist(), 'Centroid label' : centroid_labels})
    
    # Get centroid for the target cluster label
    target_centroid = df_centroids[df_centroids['Centroid label'] == target_cluster_label]['Centroid'].item()

    dists_from_target = []
    if metric == 'euclidean':
        dists_from_target = euclidean_distances(X_train, [np.asarray(target_centroid)])
    elif metric == 'manhattan':
        dists_from_target = manhattan_distances(X_train, [np.asarray(target_centroid)])
    dists_from_target = [dist[0] for dist in dists_from_target.tolist()]
    df_output['Association to cluster group: "' + target_header + '_' + str(target_cluster_label) + '"' ] = dists_from_target

    print('[Done]')

    return df_output

# Helper function for Kmeans centroids distance association calculation
def kmeans_centroids_features(df, target_header, n_clusters, target_cluster_label=1, metric='euclidean', numerical_imputer=None, scaler=None, encoder=None, remove_binary=None):
    """
    Helper function for Kmeans centroids distance association calculation in the dataset

    Arguments:
    -----------
    df : pd.dataframe, dataframe to be passed as input
    target_header : string, the header description of the column containing the cluster label
    n_clusters: int, the number of clusters specified for the Kmeans model
    metric : string, selection of 'euclidean' (minimizing sum of distances), 'manhattan' (median of distances)
    numerical_imputer : selection of 'mean', 'median', 'most_frequent', the type of imputation strategy for processing missing data
    scaler : string, selection of 'standard', 'minmax' or 'robust', type of scaler used for data processing
    encoder : selection of 'one_hot', 'label_encoding', the type of encoding method for categorical data
    remove_binary : boolean, option to remove columns containing binary values

    Returns:
    -----------
    df_output : pd.dataframe, resulting dataframe as output
    """

    # Apply the optional data transformation (imputing, scaling, encoding) if required 
    df_x, df_y = transform_data(df, target_header, numerical_imputer, scaler, encoder, remove_binary)
    df_output = df
    print('Calculating data centroids... ', end='')

    model = KMeans(n_clusters=n_clusters).fit(df_x)
    y_pred = model.predict(df_x)
    df_pred = pd.DataFrame({'Centroid label' : y_pred.tolist()})
    
    centroids = model.cluster_centers_
    centroid_labels = [model.predict([centroid])[0] for centroid in centroids]
    df_centroids = pd.DataFrame({'Centroid' : centroids.tolist(), 'Centroid label' : centroid_labels})

    # Get centroid of the cluster label assigned to each data record
    y_pred_centroids = pd.merge(df_pred, df_centroids, how='left', on='Centroid label')
    pred_centroid = y_pred_centroids['Centroid'].values.tolist()

    dists_from_target = []
    if metric == 'euclidean':
        row_count = 0
        for index, row in df_x.iterrows():
            dists_from_target.append((sum((row.values.astype(float) - pred_centroid[row_count])**2))**(1/2))
            row_count += 1
    elif metric == 'manhattan':
        row_count = 0
        for index, row in df_x.iterrows():
            dists_from_target.append(sum(abs(row.values.astype(float) - pred_centroid[row_count])))
            row_count += 1
 
    df_output['Association to nearest Kmeans cluster group'] = dists_from_target

    print('[Done]')

    return df_output

# Feature analysis with correlations
def correlations_check(df, target_header, target_label=None, numerical_imputer=None, scaler=None, encoder=None, remove_binary=None):
    """
    Helper function that outputs a table of feature correlations against a specified column in the dataset

    Arguments:
    -----------
    df : pd.dataframe, dataframe to be passed as input
    target_header : string, the header description of the column to run feature correlations against
    target_label : string, optional input if the target column is not yet encoded with binary 0 and 1 labels
    numerical_imputer : selection of 'mean', 'median', 'most_frequent', the type of imputation strategy for processing missing data
    scaler : string, selection of 'standard', 'minmax' or 'robust', type of scaler used for data processing
    encoder : selection of 'one_hot', 'label_encoding', the type of encoding method for categorical data
    remove_binary : boolean, option to remove columns containing binary values

    Returns:
    -----------
    df_correlations : pd.dataframe, resulting dataframe as output
    """

    # Apply the optional data transformation (imputing, scaling, encoding) if required 
    df_x, df_y = transform_data(df, target_header, numerical_imputer, scaler, encoder, remove_binary)

    # Check if target labels are binary 0 and 1
    
    print('Inspect target data type... ', end='')
    categorical_headers = df_y.loc[:, df_y.dtypes == object].copy()
    if len(categorical_headers.columns.tolist()) > 0:
        if len(df_y[target_header].unique()) == 2 and (0 in df_y[target_header].unique()) and (1 in df_y[target_header].unique()):
            pass
        else:
            if target_label is not None:
                # Else if column values not binary 0 and 1, proceed to encode target labels with one-hot encoding
                df_y = pd.get_dummies(df_y)

                # Select the relevant column of the specified target value as per input
                target_header = target_header + '_' + target_label
                df_y = df_y[[target_header]]
            else:
                print('Note: Target column appear to contain multiple labels. Please specify a target label.')
    print('[Done]')

    # Get the correlation values with respect to the target column
    df_x.reset_index(drop=True, inplace=True)
    df_y.reset_index(drop=True, inplace=True)
    df_correlations = pd.concat([df_x, df_y], axis=1).corr()[[target_header]]
    
    if len(df_correlations.columns.tolist()) == 1:
        print('Extracting data correlations... ', end='')
        df_correlations = df_correlations.sort_values(by=[target_header], ascending=False)
        print('[Done]')
    else:
        print('Extracting data correlations... ', end='')
        df_correlations = df_correlations.loc[:, ~df_correlations.columns.duplicated()]
        df_correlations = df_correlations.sort_values(by=[target_header], ascending=False)
        print('[Done]')
        print('Note: Potential problem: target label is not unique in correlations dataframe. Duplicates were removed.')

    # Drop the row with the index containing the original target header (i.e. drop the target label columns as correlation is relevant only for indepdent variables)
    try:
        index_labels = df_correlations.index.tolist()
        for label in index_labels:
            if target_header in label:
                df_correlations.drop(df_correlations.index[df_correlations.index.get_loc(label)], inplace=True)
    except:
        pass
    
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

# Secondary helper function for applying uniform target label distribution across train test data split
def train_test_uniform_split(df, target_header, test_size=0.3, random_state=None):
    # Split datasets w.r.t to target label to enforce uniformity of result
    df_train_1 = df[df[target_header] == 1]
    df_train_0 = df[df[target_header] == 0]
    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(df_train_1.drop([target_header], axis=1), df_train_1[[target_header]], test_size=test_size, random_state=random_state)
    X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(df_train_0.drop([target_header], axis=1), df_train_0[[target_header]], test_size=test_size, random_state=random_state)

    # Rejoin splits tests for finalizing X_train, X_test, y_train, y_test partitions
    X_train = pd.concat([X_train_0, X_train_1], axis=0)
    X_test = pd.concat([X_test_0, X_test_1], axis=0)
    y_train = pd.concat([y_train_0, y_train_1], axis=0)
    y_test = pd.concat([y_test_0, y_test_1], axis=0)
    
    # Get target labels data as arrays
    y_train = y_train[y_train.columns[0]]
    y_test = y_test[y_test.columns[0]]

    return X_train, X_test, y_train, y_test

# Kmeans clustering elbow analysis
def kmeans_elbow_plot(df, target_header, max_clusters=10):
    """
    Helper function generates a Kmeans elbow plot (for finding optimal number of Kmeans clusters) over the input dataset.

    Arguments:
    -----------
    df : pd.dataframe, dataframe to be passed as input
    target_header : string, the header description of the column containing the target label
    max_clusters : int, maximum number of clusters specified for the Kmeans models
    
    Returns:
    -----------
    Display of Kmeans elbow plot
    """

    distances_centroids = []

    df_x = df.drop([target_header], axis=1)

    K = range(1, max_clusters + 1)

    # Train the Kmeans model
    for k in K:
        model = KMeans(n_clusters=k).fit(df_x)
        model.fit(df_x)
        centroids = model.cluster_centers_
        distances_centroids.append(sum(np.min(cdist(df_x, centroids, 'euclidean'), axis=1))/df_x.shape[0])

        percent_progress = round((float(k)/float(max_clusters))*100, 2)
        if percent_progress % 1 == 0:
            print('\rData processing in progress: ' + str(int(percent_progress)) + '%', end='')

    print('')
    # Kemans elbow plot
    plt.figure(figsize=(9, 7))
    plt.plot(K, distances_centroids, color='darkblue')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distance measure from centroids')
    plt.title('Kmeans fit over varying clusters')
    plt.show()
    
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
    y = df_y
    
    # Get the encoded target labels if necessary
    # Check if target labels are binary 0 and 1
    print('Inspect target data type... ', end='')
    binary_col_headers = get_binary_headers(df_y, [target_header])
    if len(binary_col_headers) == 0:
        y = df[target_header]
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
    df_train = pd.DataFrame(X, columns=feature_headers)
    df_train[target_header] = y
    X_train, X_test, y_train, y_test = train_test_uniform_split(df_train, target_header, test_size=0.3)
    
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
def logistic_reg_features(df, target_header, target_label=None, reg_C=10, reg_norm='l2', numerical_imputer=None, scaler=None, encoder=None, remove_binary=False, n_features=None):
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
    remove_binary : boolean, option to remove columns containing binary values
    n_features : integer, number of top features to extract in running the optional recursive feature elimination function
    
    Returns:
    -----------
    df_features : pd.dataframe, resulting dataframe of model feature weights as output
    """

    # Apply the optional data transformation (imputing, scaling, encoding) if required 
    df_x, df_y = transform_data(df, target_header, numerical_imputer, scaler, encoder, remove_binary)

    feature_headers = df_x.columns
    X = df_x.values
    y = df_y
    
    # Get the encoded target labels if necessary
    # Check if target labels are binary 0 and 1
    print('Inspect target data type... ', end='')
    all_categorical_headers = df.loc[:, df.dtypes == object].columns.tolist()
    if len(all_categorical_headers) == 0:
        y = df[target_header]
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
                print('Note: Target column contains multiple labels. \nThe column is one-hot encoded and the first column of the encoded result is selected as the target label for feature influence analysis.\n')
    print('[Done]')

    # Split train and test data for model fitting
    df_train = pd.DataFrame(X, columns=feature_headers)
    df_train[target_header] = y
    X_train, X_test, y_train, y_test = train_test_uniform_split(df_train, target_header, test_size=0.3)
    
    print('Training model... no. of training examples: ' + str(X_train.shape[0]) + ', no. of features: ' + str(X_train.shape[1]) + '. ', end='')
    # Perform model training and evaluation
    model = LogisticRegression(C=reg_C, penalty=reg_norm)

    if n_features is not None:
        model_rfe = RFE(model, n_features)
        model_rfe_trained = model_rfe.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)
    print('[Done]')
    
    # Get the model performance
    if n_features is not None:
        print('Selected number of features for RPE: ' + str(model_rfe_trained.n_features_))
        df_features = pd.DataFrame({'Features' : df_train.drop([target_header], axis=1).columns, 'Support status' : model_rfe_trained.support_, 'Feature ranking' : model_rfe_trained.ranking_})
    else:
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

        # Show ROC plot
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
        
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        average_precision = round(average_precision_score(y_test, y_score), 2)

        # Show precision recall plot
        plt.figure(figsize=(9, 7))
        plt.plot(recall, precision, color='darkblue', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR plot, average precision: ' + str(average_precision))
        plt.show()
    
    return df_features

# Feature analysis with random forest model
def random_forest_features(df, target_header, target_label=None, n_trees=10, max_depth=None, min_samples_leaf=10, numerical_imputer=None, scaler=None, encoder=None, remove_binary=False, n_features=None):
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
    n_features : integer, number of top features to extract in running the optional recursive feature elimination function
    
    Returns:
    -----------
    df_features : pd.dataframe, resulting dataframe of model feature weights as output
    """

    # Apply the optional data transformation (imputing, scaling, encoding) if required 
    df_x, df_y = transform_data(df, target_header, numerical_imputer, scaler, encoder, remove_binary)

    feature_headers = df_x.columns
    X = df_x.values
    y = df_y
    
    # Get the encoded target labels if necessary
    # Check if target labels are binary 0 and 1
    print('Inspecting target data type... ', end='')
    all_categorical_headers = df.loc[:, df.dtypes == object].columns.tolist()
    if len(all_categorical_headers) == 0:
        y = df[target_header]
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
                print('Note: Target column contains multiple labels. \nThe column is one-hot encoded and the first column of the encoded result is selected as the target label for feature influence analysis.\n')
    print('[Done]')

    # Split train and test data for model fitting
    df_train = pd.DataFrame(X, columns=feature_headers)
    df_train[target_header] = y
    X_train, X_test, y_train, y_test = train_test_uniform_split(df_train, target_header, test_size=0.3)
 
    print('Training model... no. of training examples: ' + str(X_train.shape[0]) + ', no. of features: ' + str(X_train.shape[1]) + '. ', end='')
    # Perform model training and evaluation
    model = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth, min_samples_leaf=min_samples_leaf)

    if n_features is not None:
        model_rfe = RFE(model, n_features)
        model_rfe_trained = model_rfe.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)
    print('[Done]')
    
    # Get the model performance
    if n_features is not None:
        print('Selected number of features for RPE: ' + str(model_rfe_trained.n_features_))
        df_features = pd.DataFrame({'Features' : df_train.drop([target_header], axis=1).columns, 'Support status' : model_rfe_trained.support_, 'Feature ranking' : model_rfe_trained.ranking_})
    else:
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

        # Show ROC plot
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
        
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        average_precision = round(average_precision_score(y_test, y_score), 2)

        # Show precision recall plot
        plt.figure(figsize=(9, 7))
        plt.plot(recall, precision, color='darkblue', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR plot, average precision: ' + str(average_precision))
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

# Create bar plot based on the correlations of the features with respect to a target column header in the dataset.
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

# Create bar plot based on PCA features contributions with respect to a specified principal component index.
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

# Create 2D scatter of PCA biplot
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
    
# Create pie plot
def pieplot_features(df, feature_header, category_header=None, plot_size=(10, 7), title=None):
    """
    Produce a pie chart.

    Arguments:
    -----------
    df : pd.dataframe, PCA components dataframe as input data
    feature_header : string, column header of the numerical value
    category_header : string, column header of the category value
    plot_size : tuple, defines the size of the plot
    title : string, title description of the chart

    Returns:
    -----------
    Display of pie chart
    """

    # Set the plot size
    plt.figure(figsize=plot_size)
    
    # Set the plot fonts
    title_fonts, label_fonts = set_fonts()

    # Create the plot
    wedges, texts = plt.pie(df[feature_header])

    if category_header is not None:
        categories = df[category_header].tolist()
        values = df[feature_header].tolist()
        if '%' in feature_header:
            legend_labels = [str(categories[c]) + ' (' + str(round(values[c], 3)) + '%)'  for c in range(len(categories))] 
        else:
            legend_labels = [str(categories[c]) + ' (' + str(round(values[c], 3)) + ')' for c in range(len(categories))]
        plt.legend(labels=legend_labels, loc=2, bbox_to_anchor=(1.05, 1), fontsize=label_fonts['fontsize'], title=category_header)

    if title is not None:
        plt.title(title, **title_fonts)

# Create 2D distribution plot
def distplot_features(df, feature_header, target_header_value=(None, None), bin_scale=0.5, plot_size=(10, 7), xlim=(None, None), theme_style='blue', title=None, data_output=False):
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
    data_output : boolean, an option to output data points of the plot

    Returns:
    -----------
    Display of distributions plot for the selected feature(s) in the dataset
    Data points of the plot in a dataframe (optional)
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

    if data_output == True:
        df_output = pd.DataFrame({'x' : bins[:-1], 'y' : n})
        return df_output

# Create 2D distribution kde plot
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
        ax = sns.kdeplot(data=plot_data)
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
    
# Create PCA heatmap plot based on feature variance contribution across selected principal components
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
    
# Create matrix plot for correlations data
def correlation_matrix_plot(df, plot_size=(10, 7), sns_style='white', sns_context='talk', sns_cmap='plasma', annot=False, title=None):
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
        
# Create line plot
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
    
# Secondary helper function for annotating barplot labels
def annotate_bars(ax):
    bars = ax.patches
    for bar in bars:
        if bar.get_x() != bar.get_x():
            x_pos = 0
        else:
            x_pos = bar.get_x()
        if bar.get_width() != bar.get_width():
            width = 0
        else:
            width = bar.get_width()
        if bar.get_height() != bar.get_height():
            height = 0
        else:
            height = bar.get_height()
            
        ax.text(x_pos + width/2, height*1.01, round(height, 1), ha="center")

# Create bar plot
def barplot_general(df, x_header, y_header, x_label_desc=None, y_label_desc=None, ymax=None, annotation=False, hue=None, order=None, xlabel_angle=45, plot_size=(10, 7), sns_style='white', sns_context='talk', sns_palette='coolwarm_r', title=None):
    """
    Produce a general barplot.

    Arguments:
    -----------
    df : pd.dataframe, dataframe as input data to be plotted
    x_header : string, column header of the x data (independent variable)
    y_header : string, column header of the y data (dependent variable)
    x_label_desc : string, label description on x-axis
    y_label_desc : string, label description on y-axis
    ymax : float, maximum value set for y-axis
    annotation : boolean, setting for displaying annotation
    hue : string, column header for faceting the seaborn plot
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
    if order is not None:
        if order == 'descending':
            ascending_order = False
        elif order == 'ascending':
            ascending_order = True
        df = df.sort_values([y_header], ascending=ascending_order)
 
    # Create the plot
    plt.figure(figsize=plot_size)
    sns.set(context=sns_context)
    sns.set(style=sns_style)

    # Set the plot fonts
    title_fonts, label_fonts = set_fonts()
    if hue is not None:
        ax = sns.barplot(x=x_header, y=y_header, hue=hue, data=df, palette=sns_palette)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=xlabel_angle)
        plt.legend(loc=2, bbox_to_anchor=(1.05, 1), fontsize=label_fonts['fontsize'])
    else:
        ax = sns.barplot(x=x_header, y=y_header, data=df, palette=sns_palette)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=xlabel_angle)

    if title is not None:
        plt.title(title, **title_fonts)

    if x_label_desc is not None:
        plt.xlabel(xlabel=x_label_desc, **label_fonts)

    if y_label_desc is not None:
        plt.ylabel(ylabel=y_label_desc, **label_fonts)

    if ymax is not None:
        plt.ylim(bottom=0, top=ymax)

    if annotation == True:
        annotate_bars(ax)
        
# Create 2D geospatial plot
def geoplot_2d(df_geo, feature_header, scheme='quantiles', n_segment=5, plot_size=(13, 13), xlim=(None, None), ylim=(None, None), legend=True, legend_header=None, cmap='plasma', title=None):
    """
    Produce a 2D geoplot.

    Arguments:
    -----------
    df_geo : geopandas dataframe, geopandas dataframe as input data to be plotted
    feature_header : string, column header containing the numerical values as feature on the 2D geospatial plot.
    scheme : selection of in-built pandas data segmentation method
    n_segment : int, number of segments to split the data values in for plotting
    plot_size : tuple, the setting of the plot size
    xlim : tuple, defines the x-axis limits for the plot
    ylim : tuple, defines the y-axis limits for the plot
    legend : boolean, option to display the legend
    legend_header : string, optional description of the legend header
    cmap : selection of in-built matplotlib colour map options
    title : string, title description of the chart

    Returns:
    -----------
    Display of barplot
    """
     # Set the plot fonts
    title_fonts, label_fonts = set_fonts()

    # Process the plot
    if legend_header is None:
        legend_header=feature_header

    leg_kwds = {'title' : legend_header, 'loc' : 2, 'bbox_to_anchor' : (1.05, 1)}
    df_geo.plot(figsize=plot_size, column=feature_header, scheme=scheme, k=n_segment, legend=legend, legend_kwds=leg_kwds, cmap=cmap)
        
    ax = plt.gca()

    if title is not None:
        ax.set_title(title, **title_fonts)

    if xlim[0] is not None and xlim[1] is not None:
        ax.set_xlim([xlim[0], xlim[1]])
    
    if ylim[0] is not None and ylim[1] is not None:
        ax.set_ylim([ylim[0], ylim[1]])

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
def training_split(df, target_header, train_partition=0.6, dev_partition=0.2, test_partition=0.2, random_state=None):
    """
    Split the data into training, validation and test portions.

    Arguments:
    -----------
    df : pd.dataframe, dataframe to be passed as input
    target_header : string, the column with the header description of the target label
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
        X_train_dev, X_test, y_train_dev, y_test = train_test_uniform_split(df, target_header, test_size=test_partition, random_state=random_state)
        df_train_dev = pd.concat([X_train_dev, y_train_dev], axis=1)
        if dev_partition != 0:
            partition = dev_partition/(1 - test_partition)
            X_train, X_dev, y_train, y_dev = train_test_uniform_split(df_train_dev, target_header, test_size=partition, random_state=random_state)
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
        X_train_dev, X_test, y_train_dev, y_test = train_test_uniform_split(df, target_header, test_size=test_partition, random_state=random_state)
        partition = dev_partition/(1 - test_partition)
        X_train, X_dev, y_train, y_dev = train_test_uniform_split(df_train_dev, target_header, test_size=partition, random_state=random_state)

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
