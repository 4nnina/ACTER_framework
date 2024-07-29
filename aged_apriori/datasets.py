import pandas as pd
import numpy as np
import os
import datetime as dt
from datetime import timedelta
from data_classes import FitbitDataSet, AuditelDataset, ShopDataset, DiabetesDataset

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
    sleep_quality_df['timestamp'] = sleep_quality_df['timestamp'].astype('datetime64').dt.date  # crop date only
    sleep_quality_df['timestamp'] = sleep_quality_df['timestamp'] - pd.Timedelta(days=1)        # timeshift back 1 day
    sleep_quality_df = sleep_quality_df[['timestamp','overall_score']]                          # select only needed columns
    sleep_quality_df['timestamp'] = sleep_quality_df['timestamp'].astype('datetime64')          # change type to allow later merge
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


def generate_dataset_from_user(fitbit_dataset: FitbitDataSet, user_index: int, number_of_bins: int = 3) -> list[list[str]]:
    sleep_and_activity_df = generate_discretized_sleep_and_activity_df_from_user(fitbit_dataset, user_index, number_of_bins)
    
    dataset = []
    for row in sleep_and_activity_df[['light_discretized', 'moderate_discretized','heavy_discretized','rest_discretized','sleep_discretized']].itertuples():
        dataset.append([str(row.light_discretized), str(row.moderate_discretized), str(row.heavy_discretized), str(row.rest_discretized), str(row.sleep_discretized)])

    return dataset

def generate_context_from_user(fitbit_dataset: FitbitDataSet, dataset_index: int, user_index: int, context_level: int) -> list[list[str]]:
    user_data_path = fitbit_dataset.get_user_path(user_index)
    list_context_dir = os.listdir(user_data_path)

    #list of context's label
    if context_level == 4 and dataset_index == 1:
        context_types_list = ['holiday']
    elif context_level == 5 and dataset_index == 1:
        context_types_list = ['hol_we']
    else:
        context_types_list = ['week']
        if dataset_index == 1:
            if context_level > 1:
                context_types_list.append('holiday')
            if context_level > 2:
                context_types_list.append('weather')

    #user context into dataframe
    all_context_df_list = []
    for context_type in context_types_list:
        context_df_list = []
        for csv_file_name in list_context_dir:
            if csv_file_name.startswith(context_type):
                df_from_csv = pd.read_csv(user_data_path + '/' + csv_file_name)
                context_df_list.append(df_from_csv)

        context_df = pd.concat(context_df_list, axis=0, ignore_index=True)
        all_context_df_list.append(context_df)

    #merge in one dataframe
    context_df = all_context_df_list[0]
    for i in range(1,len(all_context_df_list)):
        context_df = context_df.merge(all_context_df_list[i], on='date')

    #list of context
    context = []
    for row in context_df[context_types_list].itertuples():
        context.append(list(row[1:]))
    
    return context

#AUDITEL

def generate_auditel_user_dataframe(auditel_dataset: AuditelDataset, user_index: int) -> pd.DataFrame:
    auditel_df = pd.read_csv(auditel_dataset.main_data_path)
    user_id = auditel_dataset.get_user_name(user_index)
    auditel_user_df = auditel_df[auditel_df['user_id'] == user_id]
    return auditel_user_df
  
def generate_discretized_auditel_dataset_df_from_user(auditel_dataset : AuditelDataset, user_index : int) -> pd.DataFrame:   
    auditel_user_df = generate_auditel_user_dataframe(auditel_dataset, user_index)
    auditel_user_df = auditel_user_df[auditel_user_df.genre_unique != 'DEL'] #remove under-represented category
    auditel_user_df = auditel_user_df[auditel_user_df.genre_unique != 'other']
    label_map = {
        'docu':'DC', 'fun':'FN', 'sport':'SP', 'real':'RL',
        'series':'SR', 'drama':'SR', 'fictional': 'SR',
        'action':'SR', 'thriller':'SR',
    }
    group_class_map = {
        'CHILD':'&C', 'ADULT':'&A', 'FAM_ADULT':'&T', 'FAM_CHILD':'&F'
    }
    time_slot_map = {
        'PrimeAccess': '@A', 'PrimeTime': '@T', 'DayTime': '@D',
        'LateFringe': '@L', 'EarlyFringe': '@F',  
        'EarlyMorning': '@E', 'Morning': '@M', 
        'GraveyardSlot': '@G', 
    }
    hour_map = {
        **dict.fromkeys(range(6), '*N'),
        **dict.fromkeys(range(6, 12), '*M'),
        **dict.fromkeys(range(12, 18), '*A'),
        **dict.fromkeys(range(18, 24), '*E'),
    }
    
    auditel_user_df['label'] = auditel_user_df['genre_unique'].map(lambda x : label_map[x])
    auditel_user_df['starttime'] = pd.to_datetime(auditel_user_df.starttime)
    auditel_user_df['day'] = auditel_user_df.starttime.dt.day
    auditel_user_df['day_of_week'] = auditel_user_df.day_of_week.map(lambda x : '$D' if x == 'weekday' else '$E') # #D = weekDay, #E = weekEnd
    auditel_user_df['group_class'] = auditel_user_df.group_class.map(lambda x : group_class_map[x])
    auditel_user_df['time_slot'] = auditel_user_df.time_slot.map(lambda x : time_slot_map[x])
    auditel_user_df['starttime'] = auditel_user_df.starttime.map(lambda x : hour_map[x.hour])
    
    return auditel_user_df

def generate_auditel_dataset_from_user(auditel_dataset : AuditelDataset, user_index : int, context_level : int = 0) -> list[list[str]]: 
    '''
    context levels: 
    0 -> no context
    1 -> family group
    2 -> family group + day of the week
    3 -> family group + day of the week + 4time slot 
    4 -> day of the week
    5 -> 4time slot
    6 -> family group + 4time slot
    7 -> day of the week + 4time slot
    8 -> family group + day of the week + time slot default
    '''
    auditel_user_df = generate_discretized_auditel_dataset_df_from_user(auditel_dataset, user_index)
    previous_day = auditel_user_df.iloc[0].day
    dataset = []
    for row in auditel_user_df.itertuples():
        data = [row.label, row.group_class, row.day_of_week, row.starttime, row.time_slot]
        current_day = row.day
        if current_day != previous_day:
            previous_day = current_day
            dataset.append(['G'])

        if context_level < 4:
            dataset.append(sorted(data[0:context_level + 1]))
        elif context_level < 6:
            dataset.append(sorted([data[0], data[context_level-2]]))
        elif context_level == 6:
            dataset.append(sorted([data[0], data[1], data[3]]))
        elif context_level == 7:
            dataset.append(sorted([data[0], data[2], data[3]]))
        else:
            dataset.append(sorted([data[0], data[1], data[2], data[4]]))

    return dataset

#SHOP

def generate_shop_user_dataframe(shop_dataset: ShopDataset, user_index: int) -> pd.DataFrame:
    user_id = shop_dataset.get_user_name(user_index)
    shop_df = pd.read_csv(shop_dataset.main_data_path + f'log_USER{user_id}.csv')
    shop_df['items'] = shop_df['items'].map(lambda x: x.split(','))

    return shop_df

def generate_discretized_shop_dataset_df_from_user(shop_dataset : ShopDataset, user_index : int) -> pd.DataFrame:   
    shop_user_df = generate_shop_user_dataframe(shop_dataset, user_index)
    
    label_map = {
        'soya milk':'SMILK', 'milk':'MILK', 'yogurt':'YOGURT', 'Greek yogurt': 'GYOGURT',
        'bread':'BREAD', 'breadsticks':'BREADSTICK',
        'cereals':'CER', 'cracker':'CRACKER', 'biscuits':'BISCUIT',
        'wine':'WINE', 'beer':'BEER', 'orange juice': 'JUICE', 'coca cola':'COLA',
        'vegetables': 'VEG', 'salad':'SALAD',
        'pasta':'PASTA', 'rice':'RICE',
        'fish':'FISH', 'meat':'MEAT', 'vegan burgers':'VGBURGER', 'cold cuts':'COLDC', 'jam':'JAM',
        'salt':'SALT', 'flour':'FLOUR', 'yeast':'YEAST', 'oil':'OIL', 'sugar' : 'SUGAR',
        'eggs':'EGG','frozen food': 'FROZEN',
        'toothpaste':'TOOTHP',
        'laundry detergent':'LAUDRY',
        'butter':'BUTTER', 
    }
    
    shop_user_df['label'] = shop_user_df['items'].map(lambda x : [label_map[i] for i in x])

    
    return shop_user_df

def generate_shop_dataset_from_user(shop_dataset : AuditelDataset, user_index : int) -> list[list[str]]:

    dataset_df = generate_discretized_shop_dataset_df_from_user(shop_dataset, user_index)
    dataset = dataset_df['label'].tolist()

    return dataset


#DIABETES
def discretize_event_value(dataframe_row) -> str:
    if not pd.isna(dataframe_row.duration):
        return 'E'
    elif not pd.isna(dataframe_row.dose):
        return 'B'       
    elif not pd.isna(dataframe_row.carbs):
        return 'M'
    else:
        return np.nan
    
def discretize_glucose_value(glucose_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> str:
    glucose = glucose_df[(glucose_df.ts >= start) & (glucose_df.ts <= end )]
    glucose.sort_values(by='ts', inplace=True)

    difference = glucose.val.diff()
    sum_difference = difference.sum()
    
    if difference.isna().all():
        return np.nan
    elif sum_difference < -5:
        return 'G_Decr'
    elif sum_difference > 5:
        return 'G_Incr'
    else:
        return 'G_Plateau'

def generate_diabetes_user_dataframe(diabetes_dataset: DiabetesDataset, user_index: int) -> pd.DataFrame:
    user_id = diabetes_dataset.get_user_name(user_index)
    glucose_df = pd.read_csv(diabetes_dataset.main_data_path + f'glucose_{user_id}.csv')
    bolus_df = pd.read_csv(diabetes_dataset.main_data_path + f'bolus_{user_id}.csv')
    exercise_df = pd.read_csv(diabetes_dataset.main_data_path + f'exercise_{user_id}.csv')
    meal_df = pd.read_csv(diabetes_dataset.main_data_path + f'meal_{user_id}.csv')

    glucose_df['ts'] = glucose_df.ts.apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    bolus_df['ts_begin'] = bolus_df.ts_begin.apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    bolus_df['ts_end'] = bolus_df.ts_end.apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    bolus_df['ts'] = bolus_df.ts_begin
    exercise_df['ts'] = exercise_df.ts.apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    meal_df['ts'] = meal_df.ts.apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

    
    bolus_df = bolus_df[['ts','dose', 'bwz_carb_input']]
    exercise_df = exercise_df[['ts','duration','intensity']]
    meal_df = meal_df[['ts','carbs']]
    
    return glucose_df, bolus_df, exercise_df, meal_df

def define_intervals(event_df: pd.DataFrame, min_ts: pd.Timestamp, max_ts: pd.Timestamp) -> pd.DataFrame:
    event_df = event_df.sort_values(by='ts').reset_index(drop=True)

    for e_idx in range(1, len(event_df)):
        gap_between_event = event_df.iloc[e_idx].ts - event_df.iloc[e_idx-1].ts
        event_df.loc[e_idx, 'start_glucose'] = event_df.iloc[e_idx].start_glucose - (gap_between_event/2)
        event_df.loc[e_idx-1, 'end_glucose'] = event_df.iloc[e_idx-1].end_glucose + (gap_between_event/2)

    event_df.loc[0, 'start_glucose'] = min_ts
    event_df.loc[len(event_df)-1, 'end_glucose'] = max_ts

    return event_df


def merge_into_one_dicretized_df(glucose_df: pd.DataFrame, bolus_df: pd.DataFrame, exercise_df: pd.DataFrame, meal_df: pd.DataFrame) -> pd.DataFrame:
    event_df = pd.merge(exercise_df, bolus_df, how='outer', on='ts')
    event_df = pd.merge(event_df, meal_df, how='outer', on='ts')
    event_df = event_df[event_df.ts >= glucose_df.ts.min()]
    event_df['discr_event'] = event_df.apply(lambda x: discretize_event_value(x), axis=1)
    event_df = event_df[['ts', 'discr_event']]

    event_df['start_glucose'] = event_df.ts
    event_df['end_glucose'] = event_df.ts

    event_df = define_intervals(event_df, glucose_df.ts.min(), glucose_df.ts.max())
    event_df['discr_gl'] = event_df.apply(lambda x: discretize_glucose_value(glucose_df, x.start_glucose, x.end_glucose), axis=1)
    
    return event_df[['ts', 'discr_event', 'discr_gl']]

def generate_dataset_diabetes_from_user(diabetes_dataset: DiabetesDataset, user_index: int) -> list[list[str]]:
    glucose_df, bolus_df, exercise_df, meal_df = generate_diabetes_user_dataframe(diabetes_dataset,user_index)
    discretized_interval_df = merge_into_one_dicretized_df(glucose_df, bolus_df, exercise_df, meal_df)    
    
    dataset = []
    day = discretized_interval_df.ts[0].day
    for row in discretized_interval_df.itertuples():
        if day != row.ts.day:
            dataset.append(['G'])
            day = row.ts.day
        if str(row.discr_gl) != 'nan':
            item = [str(row.discr_event), str(row.discr_gl)]
            dataset.append(item)
        
    return dataset