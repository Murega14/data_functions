import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import chardet
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, mean_absolute_error, mean_squared_error, classification_report


def load_data(file_path, encoding=None, file_types=None):
    """
    Loads data from a file into a pandas dataframe
    It determines the file type and encoding automatically

    Args:
        file_path (str): The path to the file being loaded
        encoding (str, optional): The encoding of the file being loaded. Defaults to None
        file_types: (list, optional): List of accepted file extensions. Defaults to None
            ['.csv', '.xls', '.xlsx', '.json', '.html', '.xml', '.clipboard', '.excel',
            '.hdf', '.feather', '.parquet', '.orc', '.stata', '.sas', '.spss', '.pickle', '.sql', '.gbq'].

    Returns:
        pandas.Dataframe: The loaded dataframe

    Raises:
        ValueError: If the file type is not supported or the encoding is not valid

    Examples:
        # Load a csv file
        >>> df = load_data('data.csv')

        # Load a xlsx file
        >>> df = load_data('data.xlsx')

        # Load a xls file
        >>> df = load_data('data.xls')

        # Load a json file
        >>> df = load_data('data.json')

        # Load a html file
        >>> df = load_data('data.html')

        # Load a xml file
        >>> df = load_data('data.xml')

        # Load a clipboard file
        >>> df = load_data('data.clipboard')

        # Load a excel file
        >>> df = load_data('data.excel')

        # Load a hdf file
        >>> df = load_data('data.hdf')

        # Load a feather file
        >>> df = load_data('data.feather')

        # Load a parquet file
        >>> df = load_data('data.parquet')
    """
    if file_types is None:
        file_types = ['.csv', '.xls', '.xlsx', '.json', '.html', '.xml', '.clipboard', '.excel',
                      '.hdf', '.feather', '.parquet', '.orc', '.stata', '.sas', '.spss', '.pickle', '.sql', '.gbq']

    if encoding is None:
        encoding = 'utf-8'

    try:
        with open(file_path, 'rb') as f:
            rows = f.read()
            encoding = chardet.detect(rows).get('encoding')

        if any(file_path.endswith(file_type) for file_type in file_types):
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, encoding=encoding)
            elif file_path.endswith('.xls') or file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path, encoding=encoding)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path, encoding=encoding)
            elif file_path.endswith('.html'):
                df = pd.read_html(file_path, encoding=encoding)
            elif file_path.endswith('.xml'):
                df = pd.read_xml(file_path, encoding=encoding)
            elif file_path.endswith('.clipboard'):
                df = pd.read_clipboard(encoding=encoding)
            elif file_path.endswith('.excel'):
                df = pd.read_excel(file_path, encoding=encoding)
            elif file_path.endswith('.hdf'):
                df = pd.read_hdf(file_path, encoding=encoding)
            elif file_path.endswith('.feather'):
                df = pd.read_feather(file_path, encoding=encoding)
            elif file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path, encoding=encoding)
            elif file_path.endswith('.orc'):
                df = pd.read_orc(file_path, encoding=encoding)
            elif file_path.endswith('.stata'):
                df = pd.read_stata(file_path, encoding=encoding)
            elif file_path.endswith('.sas'):
                df = pd.read_sas(file_path, encoding=encoding)
            elif file_path.endswith('.spss'):
                df = pd.read_spss(file_path, encoding=encoding)
            elif file_path.endswith('.pickle'):
                df = pd.read_pickle(file_path, encoding=encoding)
            elif file_path.endswith('.sql'):
                df = pd.read_sql(file_path, encoding=encoding)
            elif file_path.endswith('.gbq'):
                df = pd.read_gbq(file_path, encoding=encoding)
        else:
            raise ValueError(f"File type {os.path.splitext(file_path)[1]} is not supported.")
    
    except Exception as e:
        raise ValueError(f"Error loading file: {e}")
    # Check if the file extension is in the file_types list
    if os.path.splitext(file_path)[1] not in file_types:
        raise ValueError(f"File type {os.path.splitext(file_path)[1]} is not supported.")

    # Load the data based on the file extension
    if os.path.splitext(file_path)[1] == '.csv':
        df = pd.read_csv(file_path, encoding=encoding)
    elif os.path.splitext(file_path)[1] == '.xls':
        df = pd.read_excel(file_path, encoding=encoding)


def check_duplicates(df):
    """
    Checks for duplicates in all columns of a DataFrame and drops them if found.
    
    Args:
        df (pandas.DataFrame): The input DataFrame to check for duplicates.

    Returns:
        pandas.DataFrame or bool:
            - Returns a DataFrame without duplicates if duplicates are found and removed.
            - Returns False if no duplicates exist.

    This function iterates through each column in the DataFrame and checks for duplicates.
    If duplicates are found in any column, they are dropped from the DataFrame.
    If no duplicates are found in any column, it returns False.
    """

    duplicates_found = False

    # Iterate through each column in the DataFrame
    for column in df.columns:
        # Check for duplicates in the current column
        if df[column].duplicated().any():
            # Drop duplicates in the current column if found
            df.drop_duplicates(subset=column, inplace=True)
            duplicates_found = True  # Set flag to indicate duplicates were found

    # Check if duplicates were found in any column
    if duplicates_found:
        return df  # Return the modified DataFrame without duplicates
    else:
        return False  # Return False indicating no duplicates were found


def replace_missing_numerical(df, strategy='median'):
    """
    Replaces missing values in numerical columns with the mode or median

    Args:
        df (pandas.Dataframe): The input Dataframe
        strategy (str): The strategy to use for replacing missing values
            - 'mode': Fills missing values with the mode of the column
            - 'median': Fills missing values with the median of the column
            Defaults to 'median'.

    Returns:
        pandas.Dataframe: Dataframe with the missing values replaced

    Examples:
         # Replace missing values with median
        >>> df_with_median = replace_missing_numerical(df, strategy='median')

        # Replace missing values with mode
        >>> df_with_mode = replace_missing_numerical(df, strategy='mode')
    """
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

    for col in numerical_columns:
        if df[col].isnull().sum() > 0:
            if strategy == 'median':
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == 'mode':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                raise ValueError(f'Invalid strategy: {strategy}')
            
    return df

def replace_missing_categorical(df, strategy='mode'):
    """
    Replaces missing values in categorical columns with the mode

    Args:
        df (pandas.Dataframe): The input Dataframe
        strategy (str): The strategy to use for replacing missing values
            - 'mode': Fills missing values with the mode of the column

    Returns:
        pandas.Dataframe: Dataframe with the missing values replaced

    Examples:
         # Replace missing values with mode
        >>> df_with_mode = replace_missing_categorical(df)

        """
    categorical_columns = df.select_dtypes(include=['object']).columns

    for col in categorical_columns:
        if df[col].isnull().sum() > 0:
            mode_value = df[col].mode()[0]
            df[col].fillna(mode_value, inplace=True)

    return df

def convert_to_numeric(df, columns):
    """
    Converts one or multiple columns in a Dataframe to a numeric format

    Args:
        df (pandas.DataFrame): The DataFrame containing the column to convert.
        columns (str or list): The name(s) of the column(s) to convert.

    Returns:
        pandas.DataFrame: The DataFrame with the converted column(s).

    This function converts the specified column in the DataFrame to numeric format.
    It uses the pandas.to_numeric() function to convert the column to numeric format.
    If the column is already in numeric format, it is returned unchanged.

    Examples:
        #convert 'column_name' to numeric format
        >>> df_with_numeric = convert_to_numeric(df, 'column_name')

        #convert multiple columns to numeric format
        >>> df_with_numeric = convert_to_numeric(df, ['column_1', 'column_2'])
    
    """
    if isinstance(columns, str):
        columns = [columns]

    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def convert_to_boolean(df, columns):
    """
    Converts one or multiple columns in a DataFrame to a boolean format.

    Args:
        df (pandas.DataFrame): The DataFrame containing the column to convert.
        columns (str or list): The name(s) of the column(s) to convert.

    Returns:
        pandas.DataFrame: The DataFrame with the converted column(s).

    This function converts the specified column in the DataFrame to boolean format.
    It uses the pandas.to_numeric() function to convert the column to boolean format.
    If the column is already in boolean format, it is returned unchanged.

    Examples:
        #convert 'column_name' to boolean format
        >>> df_with_boolean = convert_to_boolean(df, 'column_name')

        #convert multiple columns to boolean format
        >>> df_with_boolean = convert_to_boolean(df, ['column_1', 'column_2'])
    """
    if isinstance(columns, str):
        columns = [columns]

    df[columns] = pd.to_numeric(df[columns], errors='coerce')
    df[columns] = df[columns].astype(bool)

def convert_to_datetime(df, column):
    """
    Converts a column in a DataFrame to datetime format.

    Args:
        df (pandas.DataFrame): The DataFrame containing the column to convert.
        column (str): The name of the column to convert.

    Returns:
        pandas.DataFrame: The DataFrame with the converted column.

    This function converts the specified column in the DataFrame to datetime format.
    It uses the pandas.to_datetime() function to convert the column to datetime format.
    If the column is already in datetime format, it is returned unchanged.

    Examples:
        #convert 'date_column' to datetime format
        >>> df_with_datetime = convert_to_datetime(df, 'date_column')
    """
    df[column] = pd.to_datetime(df[column], errors='coerce')
    return df
   

def plot_distribution(df, column, figure_size=(12, 8), title=None, xlabel=None):
    """
    Plots the distribution of a column in a dataframe.
    Args:
        df (pandas.DataFrame): The dataframe to plot the distribution of the column.
        column (str): The name of the column to plot the distribution of.
    Returns:
        None.
    Raises:
        None.
    Examples:
        >>> plot_distribution(df, 'column_name')
        Plots the distribution of the column 'column_name' in the dataframe 'df'.
        >>> plot_distribution(df, 'column_name', 'figure_size')
        Plots the distribution of the column 'column_name' in the dataframe 'df' with the figure size 'figure_size'.
        >>> plot_distribution(df, 'column_name', 'figure_size', 'title')
        Plots the distribution of the column 'column_name' in the dataframe 'df' with the figure size 'figure_size' and the title 'title'.
        >>> plot_distribution(df, 'column_name', 'figure_size', 'title', 'xlabel')
    """
    plt.figure(figsize=(figure_size))
    sns.displot(df[column])
    plt.xlabel(xlabel if xlabel else column)
    plt.grid(False)
    plt.title(title if title else f'Distribution of {column}')
    plt.show()


def evaluate_model(y_true, y_pred, model_type=None):
    """
    Evaluate's a model's performance based on predicted and true values

    Args:
        y_true (array-like): The true values
        y_pred (array-like): The predicted values
        model_type (str): The type of model. Options are: 'regression', 'classification', 'clustering', 'anomaly'.

    Returns:
        dict: A dictionary containing the evaluation metrics.

    This function evaluates the performance of a model based on predicted and true values.
    It returns a dictionary containing the evaluation metrics.
    The evaluation metrics depend on the type of model.
    For regression models, the metrics include 'MAE', 'MSE', 'RMSE', 'R2', 'RMSLE', 'MAPE'.
    For classification models, the metrics include 'accuracy', 'precision', 'recall', 'f1-score', 'roc-auc'.
    For clustering models, the metrics include 'silhouette-score', 'davies-bouldin-index'.
    For anomaly models, the metrics include 'AUC', 'precision', 'recall', 'f1-score'.

    Examples:
        # Evaluate regression model
        >>> metrics = evaluate_model(true_values, predicted_values, model_type='regression)

        # Evaluate a classification model
        >>>> metrics = evaluate_model(true_values, predicted_values, model_type='regression)

        $ Evaluate a model without specifying the model type
        >>> metrics = evaluate_model(true_values, predicted_values)

        # Evaluate a clustering model
        >>> metrics = evaluate_model(true_values, predicted_values, model_type='clustering)

        # Evaluate an anomaly model
        >>> metrics = evaluate_model(true_values, predicted_values)
    """

    metrics = {}

    if model_type == 'classification':
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred)
        metrics['recall'] = recall_score(y_true, y_pred)
        metrics['f1-score'] = f1_score(y_true, y_pred)
        metrics['roc-auc'] = roc_auc_score(y_true, y_pred)
        metrics['confusion-matrix'] = confusion_matrix(y_true, y_pred)
        metrics['classification-report'] = classification_report(y_true, y_pred)

    elif model_type == 'regression':
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred)
        metrics['recall'] = recall_score(y_true, y_pred)
        metrics['f1-score'] = f1_score(y_true, y_pred)
        metrics['MAE'] = mean_absolute_error(y_true, y_pred)
        metrics['MSE'] = mean_squared_error(y_true, y_pred)

    return metrics