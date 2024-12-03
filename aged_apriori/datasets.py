import pandas as pd
import numpy as np
import os
import datetime as dt
from datetime import timedelta
import sys
sys.path.append('../')
from aged_apriori.data_classes import FitbitDataSet

def generate_user_dataframes(fitbit_dataset: FitbitDataSet, user_index: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    user_data_path = fitbit_dataset.get_user_path(user_index)
    list_activity_dir = os.listdir(user_data_path)
    activity_types_list = ['light','moderate','very_active','sedentary']

    all_activities_df_list = []

    for activity_type in activity_types_list:
        activity_df_list = []
        for json_file_name in list_activity_dir:
            if json_file_name.startswith(activity_type):
                df_from_json = pd.read_json(user_data_path + '/' + json_file_name)
                activity_df_list.append(df_from_json)

        activity_df = pd.concat(activity_df_list, axis=0, ignore_index=True)
        all_activities_df_list.append(activity_df)

    light_activity_df, moderate_activity_df, heavy_activity_df, rest_df = all_activities_df_list[0], all_activities_df_list[1], all_activities_df_list[2], all_activities_df_list[3]
    sleep_quality_df = pd.read_csv(user_data_path + '/' + 'sleep_score.csv')
    return light_activity_df, moderate_activity_df, heavy_activity_df, rest_df, sleep_quality_df

def timeshift_sleep_quality_dataframe(sleep_quality_df: pd.DataFrame) -> pd.DataFrame:
    sleep_quality_df['timestamp'] = pd.to_datetime(sleep_quality_df['timestamp']).dt.date       # crop date only
    sleep_quality_df['timestamp'] = sleep_quality_df['timestamp'] - pd.Timedelta(days=1)        # timeshift back 1 day
    sleep_quality_df = sleep_quality_df[['timestamp','overall_score']]                          # select only needed columns
    sleep_quality_df['timestamp'] = pd.to_datetime(sleep_quality_df['timestamp'])               # change type to allow later merge
    return sleep_quality_df

def generate_sleep_and_activity_df(light_activity_df: pd.DataFrame, moderate_activity_df: pd.DataFrame, heavy_activity_df: pd.DataFrame, rest_df: pd.DataFrame, sleep_quality_df: pd.DataFrame) -> pd.DataFrame: 
    sleep_quality_df = timeshift_sleep_quality_dataframe(sleep_quality_df)
    
    sleep_quality_df.columns = ['date','sleep']
    light_activity_df.columns = ['date', 'light']
    moderate_activity_df.columns = ['date', 'moderate']
    heavy_activity_df.columns = ['date','heavy']
    rest_df.columns = ['date','rest']

    sleep_and_activity_df = light_activity_df.merge(moderate_activity_df, on='date').merge(heavy_activity_df, on='date').merge(rest_df, on='date').merge(sleep_quality_df, on='date')
    return sleep_and_activity_df

def discretize_values_sleep_and_activity_df(sleep_and_activity_df: pd.DataFrame, number_of_bins: int) -> pd.DataFrame:

    def jitter(a_series, noise_reduction=1000000):
        np.random.seed(33) # changing this value will change the results #42, 12 good, 33 great for pmdata
        return (np.random.random(len(a_series))*a_series.std()/noise_reduction)-(a_series.std()/(2*noise_reduction))

    value_labels = { # TODO #12 need a way to change the discretizations in real time (i.e. maybe we want 5 levels instead of 3)
        'light': ['LA_' + str(i) for i in range(1, number_of_bins + 1)],
        'moderate': ['MA_' + str(i) for i in range(1, number_of_bins + 1)],
        'heavy': ['HA_' + str(i) for i in range(1, number_of_bins + 1)],
        'rest': ['R_' + str(i) for i in range(1, number_of_bins + 1)],
        'sleep': ['ZL_' + str(i) for i in range(1, number_of_bins + 1)]
    }

    for activity_type in value_labels.keys():
        jittered_series = sleep_and_activity_df[activity_type] + jitter(sleep_and_activity_df[activity_type])   # added some jitter noise to avoid cases in which qcut has too many 0 values and has to split them into different bins
        sleep_and_activity_df[activity_type + '_discretized'] = pd.qcut(jittered_series, q=len(value_labels[activity_type]), labels=value_labels[activity_type])
    
    return sleep_and_activity_df

def generate_discretized_sleep_and_activity_df_from_user(fitbit_dataset: FitbitDataSet, user_index: int, number_of_bins: int) -> pd.DataFrame:
    light_activity_df, moderate_activity_df, heavy_activity_df, rest_df, sleep_quality_df = generate_user_dataframes(fitbit_dataset, user_index)
    sleep_and_activity_df = generate_sleep_and_activity_df(light_activity_df, moderate_activity_df, heavy_activity_df, rest_df, sleep_quality_df)
    sleep_and_activity_df = discretize_values_sleep_and_activity_df(sleep_and_activity_df, number_of_bins)
    return sleep_and_activity_df

def generate_context_from_user(fitbit_dataset: FitbitDataSet, user_index: int, context_level: int, thresold_anomaly = None, time_steps_anomaly = None, crop = False) -> list[list[str]]:
    user_data_path = fitbit_dataset.get_user_path(user_index)
    list_context_dir = os.listdir(user_data_path)

    #list of context's label
    if context_level == 4:
        context_types_list = ['holiday']
    elif context_level == 5:
        context_types_list = ['hol_we']
    elif context_level == 6:
        context_types_list = ['anomalies']
    else:
        context_types_list = ['week']
        if context_level > 1:
            context_types_list.append('holiday')
        if context_level > 2:
            context_types_list.append('weather')

    #user context into dataframe
    if context_level != 6:
        all_context_df_list = []
        for context_type in context_types_list:
            context_df_list = []
            for csv_file_name in list_context_dir:
                if csv_file_name.startswith(context_type):
                    df_from_csv = pd.read_csv(user_data_path + '/' + csv_file_name)
                    context_df_list.append(df_from_csv)

            context_df = pd.concat(context_df_list, axis=0, ignore_index=True)
            context_df['date'] = pd.to_datetime(pd.to_datetime(context_df['date']).dt.date)
            all_context_df_list.append(context_df)

        #merge in one dataframe
        context_df = all_context_df_list[0]
        for i in range(1,len(all_context_df_list)):
            context_df = context_df.merge(all_context_df_list[i], on='date')
    else:
        if crop == 280:
            context_df = pd.read_csv(f'../anomaly/experiments/{thresold_anomaly}thresh_{time_steps_anomaly}timesteps/anomalies_{fitbit_dataset.get_user_name(user_index)}Crop280.csv')
            context_df['date'] = pd.to_datetime(pd.to_datetime(context_df['date']).dt.date)
        elif crop:
            context_df = pd.read_csv(f'../anomaly/experiments/{thresold_anomaly}thresh_{time_steps_anomaly}timesteps/anomalies_{fitbit_dataset.get_user_name(user_index)}Crop.csv')
            context_df['date'] = pd.to_datetime(pd.to_datetime(context_df['date']).dt.date)
        else:
            context_df = pd.read_csv(f'../anomaly/experiments/{thresold_anomaly}thresh_{time_steps_anomaly}timesteps/anomalies_{fitbit_dataset.get_user_name(user_index)}.csv')
            context_df['date'] = pd.to_datetime(pd.to_datetime(context_df['date']).dt.date)

    return context_df, context_types_list

def generate_dataset_from_user(fitbit_dataset: FitbitDataSet, user_index: int, number_of_bins: int = 3, context_level: int = 0, thresold_anomaly = None, time_steps_anomaly = None, crop = False) -> list[list[str]]:
    sleep_and_activity_df = generate_discretized_sleep_and_activity_df_from_user(fitbit_dataset, user_index, number_of_bins)
    
    if context_level > 0:
        context_df, context_col_name = generate_context_from_user(fitbit_dataset, user_index, context_level, thresold_anomaly,time_steps_anomaly, crop )
        sleep_and_activity_df = sleep_and_activity_df.merge(context_df, on='date', how='inner')
    else:
        context_col_name = None

    sleep_and_activity_df.sort_values(by='date', inplace=True, ascending=True, ignore_index=True)

    dataset = []
    previous_date = sleep_and_activity_df['date'][0] - timedelta(days=1)
    for row in sleep_and_activity_df.itertuples():
        curr_date = row.date
        if curr_date != previous_date + timedelta(days=1):
            dataset.append(['G'])
        else:
            activity_row = []
            if context_col_name is not None:
                for ctx in context_col_name:
                    ctx_label = str(getattr(row, ctx))
                    if ctx_label != 'noVALUE' and ctx_label != 'N_0':
                        activity_row.append(ctx_label)

            activity_row += [str(row.light_discretized), str(row.moderate_discretized), str(row.heavy_discretized), str(row.rest_discretized), str(row.sleep_discretized)]
            
            dataset.append(activity_row)
        
        previous_date = curr_date

    if crop == 280 and context_level != 6:
        dataset = dataset[-280:]

    if crop is True and context_level != 6:
        dataset = dataset[-250:]


    return dataset