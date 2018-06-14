# This module is made to aid data analysis by providing commonly used functions
# for inspecting and analysing a dataset.
#
# It may be helpful for:
# - Inspecting a set of data points for its distribution characterstic
# - Inspecting a set of data points for unique labels and their counts
# - Outputing transformed feature data via common feature engineering methods
# - Outputing transformed target label data as binary for One Vs All classification tasks

# Setup data processing dependencies
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold, ShuffleSplit
from skimage.io import imread, imshow
from skimage.transform import rescale, resize, downscale_local_mean

# ===============================
# === Inspection and analysis ===

# Get dataframe containing information about the unique labels within the dataset column/array.
def datasets_overview(df_list, df_names):
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
    df_list : pd.dataframe, list of dataframes as input.

    Returns:
    -----------
    df_common : pd.dataframe, dataframe detailing the common column features.
    """
    common_col_names = []
    for df in df_list:
        if len(common_col_names) == 0:
            common_col_names = df.columns.tolist()
        else:
            common_col_names = list(
                set(common_col_names) & set(df.columns.tolist()))

    df_common = pd.DataFrame(
        {'Features(s) common between the datasets': common_col_names})
    return df_common

# Compare and find different column features between a main and other given dataframes.
def compare_difference(df_other, df_train):
    """
    Helper function that compares a main dataframe (typically the training dataset) with a different dataframe and output their different column features with respect to the other dataframe.

    Arguments:
    -----------
    df_list : pd.dataframe, list of dataframes to be passed in as input.

    Returns:
    -----------
    df_difference : pd.dataframe, dataframe detailing the different column features.
    """
    train = df_train.columns.tolist()
    other = df_other.columns.tolist()
    col_names_union = list(set().union(other, train))
    col_names_intersect = list(set(other) & set(train))
    diff_col_names = list(
        set(col_names_union) - set(col_names_intersect) - set(df_train.columns.tolist()))

    df_difference = pd.DataFrame(
        {'Features(s) not found in the training dataset': diff_col_names})
    return df_difference


def unique_values(df, target_header):
    """
    Helper function that outputs a table containing the unique label, the coresponding entries count and the relatively representation of the data in the dataset.

    Arguments:
    -----------
    df : pd.dataframe, dataframe to be passed in as input.
    target_header : string, the column header description to apply analysis on.

    Returns:
    -----------
    df_unique : pd.dataframe, resulting dataframe as output.
    """
    target_header_desc = target_header
    target_header_no = '(' + 'Unique values: ' + \
        str(df[target_header].nunique()) + ')'

    # Create unique value result dataframe.
    df_unique = pd.DataFrame(
        {
            str(target_header_no) + ' ' + target_header_desc: df[target_header].value_counts().index.values,
            'Number of entries': np.asarray(df[target_header].value_counts())
        }
    )

    # Relative prevalence of the data representation amongst all other categories.
    # (expressed in percentage proportion)
    label_count_sum = df_unique['Number of entries'].sum()
    df_unique['Relative representation %'] = 100 * \
        df_unique['Number of entries']/label_count_sum

    return df_unique


def relative_representation(df, header_list, min_threshold=5):
    """
    Helper function that outputs a summary table of categorical values where their relative prevelance is less than a given threshold (min_threshold).

    Arguments:
    -----------
    df : pd.dataframe, dataframe to be passed as input.
    header_list : list, list of columns headers to be passed in as input.
    min_threshold : float, the minimum threshold where the categorical value prevalence is considered to be adequately represented.

    Returns:
    -----------
    df_relative_rep : pd.dataframe, resulting dataframe as output.
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


def missing_values_table(df):
    """
    Helper function that outputs a summary table of the proportion of data that are missing with respect to each column.

    Arguments:
    -----------
    df : pd.dataframe, dataframe to be passed as input.

    Returns:
    -----------
    mis_val_table_ren_columns : pd.dataframe, resulting dataframe as output.
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

# Perform PCA and output scatter plot


def check_pca_scatter(df, scaler='standard', components=2, target_label=None):
    """
    Produce a PCA scattermap after applying scaler functions using Sklearn python library.

    Arguments:
    -----------
    df : pd.dataframe, dataframe of the input data
    scaler : string, selection of 'standard', 'minmax' or 'robust', type of scaler used for data processing
    components : integer, number of PCA components
    annot : boolean of True or False, display or not display value annotations on the heatmap

    Returns:
    -----------
    df_x_pca : pd.dataframe, dataframe containing PCA transform values
    """
    # Apply scaler to data
    if scaler == 'standard':
        scaler = StandardScaler()
        scaler.fit(df)
    elif scaler == 'minmax':
        scaler = MinMaxScaler()
        scaler.fit(df)
    else:
        scaler = RobustScaler()
        scaler.fit(df)

    # Fit data to PCA transformation
    df_pca = scaler.transform(df)

    # Get the top 2 principal components
    pca = PCA(n_components=int(components))
    pca.fit(df_pca)
    x_pca = pca.transform(df_pca)

    # Set PCA components as per input
    if components == 2:
        column_headers = ['1st principal component', '2nd principal component']
    elif components == 3:
        column_headers = ['1st principal component',
                          '2nd principal component', '3rd principal component']
    elif components == 4:
        column_headers = ['1st principal component', '2nd principal component',
                          '3rd principal component', '4th principal component']
    else:
        print('Warning: number of components argument entered is less than 2 or greater than 4. Thus, components outputs have been downscaled to 4 components.')
        column_headers = ['1st principal component', '2nd principal component',
                          '3rd principal component', '4th principal component']

    df_x_pca = pd.DataFrame(data=x_pca, columns=column_headers)
    df_x_pca = pd.concat([df_x_pca, df[target_label]], axis=1)

    # Plot the PCA scatter plot
    g_pca_scatter = sns.lmplot(data=df_x_pca, x='1st principal component', y='2nd principal component',
                               hue=target_label, fit_reg=False, palette='plasma', size=7, aspect=1.5)

    return df_x_pca

# Perform PCA and output heatmap


def check_pca_heatmap(df, scaler='standard', components=2, annot=False):
    """
    Produce a PCA heatmap after applying scaler functions using Sklearn python library.

    Arguments:
    -----------
    df : pd.dataframe, dataframe of the input data
    scaler : string, selection of 'standard', 'minmax' or 'robust', type of scaler used for data processing
    components : integer, number of PCA components
    annot : boolean, choice of true/false for display or not display value annotations on the heatmap

    Returns:
    -----------
    df_comp : pd.dataframe, dataframe containing principle component values
    """
    # Apply scaler to data
    if scaler == 'standard':
        scaler = StandardScaler()
        scaler.fit(df)
    elif scaler == 'minmax':
        scaler = MinMaxScaler()
        scaler.fit(df)
    else:
        scaler = RobustScaler()
        scaler.fit(df)

    # Fit data to PCA transformation
    df_pca = scaler.transform(df)

    # Get the top 2 principal components
    pca = PCA(n_components=int(components))
    pca.fit(df_pca)

    # Set PCA components as per input
    if components == 2:
        column_headers = ['1st principal component', '2nd principal component']
    elif components == 3:
        column_headers = ['1st principal component',
                          '2nd principal component', '3rd principal component']
    elif components == 4:
        column_headers = ['1st principal component', '2nd principal component',
                          '3rd principal component', '4th principal component']
    else:
        print('Warning: number of components argument entered is less than 2 or greater than 4. Thus, components outputs have been downscaled to 4 components.')
        column_headers = ['1st principal component', '2nd principal component',
                          '3rd principal component', '4th principal component']

    # Plot the PCA heatmap
    feature_headings = df.columns.values.tolist()
    df_comp = pd.DataFrame(pca.components_, columns=feature_headings)
    df_comp = df_comp.set_index([column_headers])
    plt.figure(figsize=(15, 7))
    g_pca_heatmap = sns.heatmap(data=df_comp, annot=annot, cmap='plasma')

    return df_comp


def create_dataframe(file_name='unknown'):
    """
    Import original data for analysis.

    Arguments:
    -----------
    file_name : string, name of the data file (excludes extension descriptions like '.csv' or '.xlsx')

    Returns:
    -----------
    df_original : pd.dataframe, dataframe of the original dataset
    """

    base_dir = os.path.dirname(os.path.realpath('__file__'))
    data_dir = os.path.join(base_dir, 'Data')
    file_dir_csv = os.path.join(data_dir, file_name + '.csv')

    file_dir_xlsx = os.path.join(data_dir, file_name + 'xlsx')
    df_original = None

    try:
        df_original = pd.read_csv(file_dir_csv, encoding='cp1252')
        print('Status: Data imported')
        print('\nData preview:')
        print(df_original.head())
        print('\nContent summary:')
        print(df_original.describe())
    except:
        try:
            df_original = pd.read_excel(file_dir_xlsx, sheet_name='sheet1')
            print('Status: Data imported')
            print('\nData preview:')
            print(df_original.head())
            print('\nContent statistics:')
            print(df_original.describe())
        except:
            print('Status: No data found!')

    return df_original


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
