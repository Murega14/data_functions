import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import chardet
import tabula
import geocoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, mean_absolute_error, mean_squared_error, classification_report
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score


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

def convert_pdf_to_csv(pdf_file, csv_file, password=None):
    """
    Converts a PDF file into a CSV file.

    Args:
        pdf_file (str): path to the input PDF  file
        csv_file (str): path to save the output CSV file
        password (str): Password for the PDF file if its encrypted. Defaults to None
    
    Returns:
        None.

    This function extracts tables from a PDF file using tabula and saves the extracted tables as CSV files.
    """
    try:
        if password:
            tables = tabula.read_pdf(pdf_file, output_format="csv", password=password)
        else:
            tables = tabula.read_pdf(pdf_file, output_format="csv")

        # concatenating all extracted tables into one dataframe
        df = pd.concat(tables)

        df.to_csv(csv_file, index=False)

        print(f"Table saved to {csv_file}")

    except Exception as e:
        print(f"Error converting PDF to CSV: {e}")  

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

def extract_date_month(df, columns):
    """
    Extracts date and month from specified columns in a Dataframe

    Args:
        df (pandas.Datframe): The Dataframe containing the columns to extract date and month
        columns (str or list): The name(s) of the columns to extract date and month from

    Returns:
        pandas.DataFrame: The Dataframe with the extracted date and month columns

    This function extracts the date and month from the specified columns in the Dataframe.
    It uses the pandas.to_datetime() function to convert the columns to datetime format.
    It then extracts the date and month from the datetime objects using the pandas.DatetimeIndex.date and pandas.DatetimeIndex.month attributes.
    The extracted date and month columns are added to the Dataframe as new columns.

    Examples:
        #extract date and month from 'date_column'
        >>> df_with_date_month = extract_date_month(df, 'date_column')

        #extract date and month from multiple columns
        >>> df_with_date_month = extract_date_month(df, ['date_column_1', 'date_column_2'])
    """
    if isinstance(columns, str):
        columns = [columns]

    for col in columns:
        if df[col].dtype != 'datetime64[ns]':
            df[col] = pd.to_datetime(df[col], errors='coerce')
        df[f'{col}_Date_Month'] = df[col].dt.strftime('%d-%m')

    return df
   
def extract_day_name(df, columns):
    """
    Extracts day name from specified columns in a Dataframe

    Args:
        df (pandas.Datframe): The Dataframe containing the columns to extract day name
        columns (str or list): The name(s) of the columns to extract day name from

    Returns:
        pandas.DataFrame: The Dataframe with the extracted day name columns

    This function extracts the day name from the specified columns in the Dataframe.
    It uses the pandas.to_datetime() function to convert the columns to datetime format.
    It then extracts the day name from the datetime objects using the pandas.DatetimeIndex.day_name attribute.
    The extracted day name columns are added to the Dataframe as new columns.

    Examples:
        #extract day name from 'date_column'
        >>> df_with_day_name = extract_day_name(df, 'date_column')

        #extract day name from multiple columns
        >>> df_with_day_name = extract_day_name(df, ['date_column_1', 'date_column_2'])
    """
    if isinstance(columns, str):
        columns = [columns]

    for col in columns:
        if df[col].dtype != 'datetime64[ns]':
            df[col] = pd.to_datetime(df[col], errors='coerce')
        df[f'{col}_Day_Name'] = df[col].dt.day_name()

    return df

def add_longitude_latitude(df, column):
    """
    Adds longitude and latitude columns to a DataFrame

    Args:
        df (pandas.DataFrame): The DataFrame to add the longitude and latitude columns to
        column (str): The name of the column to add the longitude and latitude columns from

    Returns:
        pandas.DataFrame: The DataFrame with the added longitude and latitude columns

    This function adds two new columns to the DataFrame: 'Longitude' and 'Latitude'.
    The 'Longitude' column contains the longitude value extracted from the 'column' column.
    The 'Latitude' column contains the latitude value extracted from the 'column' column.

    Examples:
        #add longitude and latitude columns to 'column_name' column
        >>> df_with_longitude_latitude = add_longitude_latitude(df, 'column_name')
    """
   #initializing the geocoder object
    geolocator = geocoder.Nominatim(user_agent="my_app")
    df['longitude'] = []
    df['latitude'] = []

    for index, row in df.iterrows():
        state_name = row[column]
        location = geolocator.geocode(state_name)

        if location:
            longitude = location.longitude
            latitude = location.latitude
        else:
            longitude = None
            latitude = None

        df.loc[index, 'longitude'] = longitude
        df.loc[index, 'latitude'] = latitude

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

def plot_bar(df, x, y, figure_size=(12, 8), title=None, xlabel=None, ylabel=None):
    """
    Plots a bar chart from two columns in a dataframe.
    Args:
        df (pandas.DataFrame): The dataframe to plot the bar chart from.
        x (str): The name of the column to use as the x-axis data.
        y (str): The name of the column to use as the y-axis data.
        figure_size (tuple, optional): Size of the figure (width, height). Defaults to (12, 8).
        title (str, optional): Title for the plot. Defaults to None.

    Returns:
        None.
    Raises:
        None.
    Examples:
        >>> plot_bar(df, 'x_column', 'y_column')
        Plots a bar chart from the 'x_column' column in the dataframe 'df' and the 'y_column' column.
        >>> plot_bar(df, 'x_column', 'y_column', 'figure_size')
        Plots a bar chart from the 'x_column' column in the dataframe 'df' and the 'y_column' column with the figure size 'figure_size'.
        >>> plot_bar(df, 'x_column', 'y_column', 'figure_size', 'title')

    """
    plt.figure(figsize=(figure_size))
    ax = sns.barplot(x=x, y=y, data=df)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.grid(False)
    plt.xticks(rotation=0)
    plt.xlabel(xlabel if xlabel else x)
    plt.ylabel(ylabel if ylabel else y)
    plt.title(title if title else f'Bar Chart of {x} vs {y}')
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
        >>>> metrics = evaluate_model(true_values, predicted_values, model_type='classification')

        $ Evaluate a model without specifying the model type
        >>> metrics = evaluate_model(true_values, predicted_values)

        # Evaluate a clustering model
        >>> metrics = evaluate_model(true_values, predicted_values, model_type='clustering)

        # Evaluate an anomaly model
        >>> metrics = evaluate_model(true_values, predicted_values, model_type='anomaly')
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
    
    elif model_type == 'clustering':
        metrics['silhouette'] = silhouette_score(y_true, y_pred)
        metrics['davies-bouldin'] = davies_bouldin_score(y_true, y_pred)
        metrics[' calinski_harabasz_score'] = calinski_harabasz_score(y_true, y_pred)

    elif model_type == 'anomaly':
        y_pred = np.array(y_pred) >= 0.5
        metrics['AUC'] = roc_auc_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred)
        metrics['recall'] = recall_score(y_true, y_pred)
        metrics['f1-score'] = f1_score(y_true, y_pred)

    else:
        print("Invalid model type provided. Please specify 'regression', 'classification', 'clustering', or 'anomaly'")

    return metrics