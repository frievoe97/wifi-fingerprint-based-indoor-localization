import pandas as pd


def read_csv(file_path):
    """
    Reads a CSV file and returns a DataFrame.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        print("CSV file successfully read.")
        return df
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None


def determine_group_columns(df):
    """
    Determines the group columns by excluding specific columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    list: The list of columns to group by.
    """
    excluded_columns = ['correct', 'duration', 'distance', 'predict_room', 'room_id', 'room_name', 'measurement_id',
                        'device_id']
    group_columns = [col for col in df.columns if col not in excluded_columns]
    return group_columns


def aggregate_data(df, group_columns, new_pattern=False):
    """
    Aggregates data based on the specified group columns and calculates the correct percentage or percentages of True, False, and Not False.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    group_columns (list): The list of columns to group by.
    new_pattern (bool): Flag to determine which aggregation pattern to use.

    Returns:
    pd.DataFrame: The aggregated DataFrame with additional columns.
    """
    group_columns = ['room_name'] + group_columns

    if new_pattern:
        # Calculate percentages of True, False, and Not False
        df['True_percent'] = df.groupby(group_columns)['correct'].transform(lambda x: (x == "True").sum() / len(x) * 100)
        df['False_percent'] = df.groupby(group_columns)['correct'].transform(
            lambda x: (x == "False").sum() / len(x) * 100)
        df['Not_False_percent'] = df.groupby(group_columns)['correct'].transform(
            lambda x: (x == "Not False").sum() / len(x) * 100)

        aggregated_data = df.groupby(group_columns).agg({
            'True_percent': 'mean',
            'False_percent': 'mean',
            'Not_False_percent': 'mean',
            'measurement_id': 'nunique',
            'distance': 'mean',
            'duration': 'mean'
        }).reset_index()

    else:
        # Calculate correct percentage
        df['correct_percent'] = df.groupby(group_columns)['correct'].transform('mean') * 100

        room_count = df.groupby(group_columns).size().reset_index(name='room_count')

        # Aggregate the data
        aggregated_data = df.groupby(group_columns).agg({
            'correct_percent': 'mean',
            'measurement_id': 'nunique',
            'distance': 'mean',
            'duration': 'mean'
        }).reset_index()

        aggregated_data = aggregated_data.merge(room_count, on=group_columns, how='left')

    return aggregated_data


def aggregate_correct_percent_by_parameters(df, group_columns, new_pattern=False):
    """
    Aggregates the correct percentages or new pattern percentages by parameter combinations.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    group_columns (list): The list of columns to group by.
    new_pattern (bool): Flag to determine which aggregation pattern to use.

    Returns:
    pd.DataFrame: The DataFrame with aggregated percentages.
    """
    if new_pattern:
        grouped_by_parameters = df.groupby(group_columns).agg({
            'True_percent': 'mean',
            'False_percent': 'mean',
            'Not_False_percent': 'mean'
        }).reset_index()
    else:
        grouped_by_parameters = df.groupby(group_columns).agg({
            'correct_percent': 'mean'
        }).reset_index()

    return grouped_by_parameters

def remove_constant_columns(df):
    """
    Removes columns from the DataFrame where all the values are the same.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The DataFrame with constant columns removed.
    """
    nunique = df.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    return df.drop(cols_to_drop, axis=1)


def measurements_per_room(df):
    """
    Calculates the number of measurements per room.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: A DataFrame with the number of measurements per room.
    """
    measurements = df.groupby('room_name')['measurement_id'].nunique().reset_index()
    measurements.columns = ['room_name', 'num_measurements']
    return measurements

def print_full_df(df):
    """
    Prints the entire DataFrame without truncation.

    Parameters:
    df (pd.DataFrame): The DataFrame to print.
    """
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)

def replace_values_based_on_condition(df, column_to_check, value_to_check, column_to_replace, replacement_value):
    """
    Replaces values in a specified column based on a condition in another column.

    :param df: The pandas DataFrame to be processed.
    :param column_to_check: The column where the condition is checked.
    :param value_to_check: The value to check for in the column_to_check.
    :param column_to_replace: The column where values will be replaced if the condition is met.
    :param replacement_value: The value to set in the column_to_replace when the condition is met.
    :return: The modified pandas DataFrame.
    """
    # Check condition and replace values
    df.loc[df[column_to_check] == value_to_check, column_to_replace] = replacement_value
    return df