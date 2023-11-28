import pandas as pd
import numpy as np
import seaborn as sns
import geocoder
import output as out


def test_load_data():
    file_path = 'test_data.csv'
    df = out.load_data(file_path)
    assert isinstance(df, pd.DataFrame)


def test_convert_pdf_to_csv():
    pdf_file = 'test_data.pdf'
    csv_file = 'test_data.csv'
    out.convert_pdf_to_csv(pdf_file, csv_file)
    df = pd.read_csv(csv_file)
    assert isinstance(df, pd.DataFrame)


def test_check_duplicates():
    data = {'col1': [1, 2, 3, 4, 1, 2, 3], 'col2': ['a', 'b', 'c', 'd', 'a', 'b', 'c']}
    df = pd.DataFrame(data)
    assert out.check_duplicates(df) is True

    df = pd.DataFrame(data, index=range(len(data)))
    assert out.check_duplicates(df) is False


def test_replace_missing_numerical():
    data = {'col1': [1, 2, np.nan, 4], 'col2': [5, 6, 7, np.nan]}
    df = pd.DataFrame(data)
    df_no_duplicates = out.replace_missing_numerical(df)
    assert df_no_duplicates.isnull().sum().sum() == 0


def test_replace_missing_categorical():
    data = {'col1': ['a', 'b', np.nan, 'd'], 'col2': ['e', 'f', 'g', np.nan]}
    df = pd.DataFrame(data)
    df_no_duplicates = out.replace_missing_categorical(df)
    assert df_no_duplicates.isnull().sum().sum() == 0


def test_convert_to_numeric():
    data = {'col1': ['1', '2', '3', '4'], 'col2': ['5', '6', '7', '8']}
    df = pd.DataFrame(data)
    df_converted = out.convert_to_numeric(df, ['col1', 'col2'])
    assert all(df_converted.dtypes == 'int64')


def test_convert_to_boolean():
    data = {'col1': [1, 0, 1, 0], 'col2': [0, 1, 0, 1]}
    df = pd.DataFrame(data)
    df_converted = out.convert_to_boolean(df, ['col1', 'col2'])
    assert all(df_converted.dtypes == 'bool')


def test_convert_to_datetime():
    data = {'col1': ['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04']}
    df = pd.DataFrame(data)
    df_converted = out.convert_to_datetime(df, 'col1')
    assert all(df_converted.dtypes == 'datetime64[ns]')


def test_extract_date_month():
    data = {'col1': ['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04']}
    df = pd.DataFrame(data)
    df_converted = out.extract_date_month(df, 'col1')
    assert all(df_converted['col1_Date_Month'].apply(lambda x: x[0] == '0' and x[1] in '123'))


def test_extract_day_name():
    data = {'col1': ['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04']}
    df = pd.DataFrame(data)
    df_converted = out.extract_day_name(df, 'col1')
    assert all(df_converted['col1_Day_Name'].apply(lambda x: x in ['Friday', 'Saturday', 'Sunday', 'Monday']))


def test_add_longitude_latitude():
    data = {'col1': ['New York', 'Los Angeles', 'Chicago', 'Houston']}
    df = pd.DataFrame(data)
    df_converted = out.add_longitude_latitude(df, 'col1')
    assert all(df_converted.columns == ['col1', 'longitude', 'latitude'])


def test_plot_distribution():
    data = {'col1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    df = pd.DataFrame(data)
    out.plot_distribution(df, 'col1')


def test_plot_bar():
    data = {'col1': ['a', 'b', 'c', 'd'], 'col2': [1, 2, 3, 4]}
    df = pd.DataFrame(data)
    out.plot_bar(df, 'col1', 'col2')


def test_evaluate_model():
    y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    y_pred = [0, 1, 0, 1, 0, 1, 0, 1]
    metrics = out.evaluate_model(y_true, y_pred, model_type='classification')
    assert all(metric in metrics for metric in ['accuracy', 'precision', 'recall', 'f1-score', 'roc-auc', 'confusion-matrix', 'classification-report'])