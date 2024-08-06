from typing import Tuple
from numpy import number
import pandas as pd
import time
import yaml
from data_classes import FitbitDataSet, TemporalEvent, CustomUsersFitbitDataset, PMDataFitbitDataset
from datasets import generate_dataset_from_user, generate_context_from_user
from experiments import filter_rules, get_rules_from_user, search_rules, sort_rules
from utils import call_aged_apriori, call_classic_apriori, clean_context, temporal_from_strings, wrapper_function
from types import FunctionType
from profilehooks import profile

datasets = [PMDataFitbitDataset, CustomUsersFitbitDataset]

#FITBIT DECORATOR
def get_validation_params_decorator(func: FunctionType) -> FunctionType:
    def inner() -> None:
        activity_types = ['HA_', 'LA_', 'MA_', 'R_', 'ZL_']

        # Load config from YAML file
        with open("config.yaml", "r") as fp:
            args = yaml.safe_load(fp)

        print('\nConfig loaded\n')

        dataset_index = args['dataset_index']
        activity_type_index = args['activity_type_index']
        activity_type = activity_types[activity_type_index]
        activity_value = args['activity_value']
        context_level = args['context_level']
        temporal_window = args['temporal_window']
        number_of_bins = args['number_of_bins']
        min_support = args['min_support']
        min_confidence = args['min_confidence']
        exponential_decay = args['exponential_decay']
        
        func(datasets[dataset_index](), dataset_index, activity_type, activity_value, context_level, temporal_window, number_of_bins,  min_support, min_confidence, exponential_decay)
    
    return inner

@get_validation_params_decorator
def validate_rules(fitbit_dataset: FitbitDataSet, dataset_index:int, activity_type: str, activity_value: int, context_level:int, temporal_window: int, number_of_bins: int, min_support: float, min_confidence: float, exponential_decay: bool) -> None:
    results = []
    df_index = []

    for user_index in range(len(fitbit_dataset.users_list)):
        user_name = fitbit_dataset.get_user_name(user_index)
        print(f'Validating experiments for {user_name}')
      
        dataset = generate_dataset_from_user(fitbit_dataset, user_index, number_of_bins)
        
        if context_level > 0:
            context = generate_context_from_user(fitbit_dataset, dataset_index, user_index, context_level)
            dataset = list(map(lambda x,y : y if (x[0] == 'noVALUE') else x+y, context, dataset))

        len_dataset = len(dataset)
        print(f'DATASET SIZE: {len_dataset}')

        #crop dataset
        if len_dataset > 150:
            crop_index = int(input(f'Crop dataset to (default = {len_dataset}): ') or len_dataset)
            dataset = dataset[-crop_index:]
            len_dataset = len(dataset)
            print(f'DATASET SIZE: {len_dataset}')
 
        if min_support == -1:
            min_support = ((140 - len_dataset)/10) * 0.001 + 0.04
        
        default_test_pct = 20 
        test_pct = int(input(f'Test % (default = {default_test_pct}): ') or default_test_pct) 
        if test_pct == 0: # to skip some problematic users
            continue

        df_index.append(user_name)
        test_index = int(len_dataset * (test_pct * 0.01))

        #train-test split
        train_dataset = dataset[:-test_index]
        test_dataset = dataset[-test_index:]

        #generate rules from training dataset with aged apriori
        aged_time = time.time()
        aged_rules = wrapper_function(train_dataset, activity_type=activity_type, activity_value=activity_value, 
                                apriori_function=call_aged_apriori, temporal_window=temporal_window, 
                                min_support=min_support, min_confidence=min_confidence, exponential_decay=exponential_decay)
        filtered_aged_rules = aged_rules        #filter_rules(aged_rules) # useless since we are matching with queries that don't have sleep in antecedent?
        aged_time = time.time() - aged_time

        #generate rules from training dataset with classic apriori
        classic_time = time.time()
        classic_rules = wrapper_function(train_dataset, activity_type=activity_type, activity_value=activity_value, 
                                apriori_function=call_classic_apriori, temporal_window=temporal_window, 
                                min_support=min_support, min_confidence=min_confidence)
        filtered_classic_rules = classic_rules  #filter_rules(classic_rules)
        classic_time = time.time() - classic_time

        #generate test queries
        query_range_end = 0
        query_range_start = query_range_end + temporal_window + 1
        aged_results = []
        classic_results = []
        while query_range_start <= test_index:   
            test_query = []
            activity_value_test_string = None
            counter = 1
            gap = False

            test_dataset_in_range = test_dataset[-query_range_start:-query_range_end]
            if query_range_end == 0:
                test_dataset_in_range = test_dataset[-query_range_start:]

            for list_of_strings in test_dataset_in_range:  
                if gap:             #found gap, reset test_query
                    test_query = []
                    gap = False

                if 'G' in list_of_strings:
                    gap = True
                else:
                    current_day = list_of_strings[:-1]
                    timeloc = temporal_window + 1 - counter 
                    list_of_activities = temporal_from_strings([current_day], timeloc=timeloc)[0]
                    test_query = test_query + list_of_activities
                    activity_value_test_string = [x for x in list_of_strings if x.startswith(activity_type)][0]
                counter += 1

            if gap:
                query_range_end += 1
                query_range_start = query_range_end + temporal_window + 1
                continue
            
            print(f'Query: {test_query}')
            activity_value_test = TemporalEvent(activity_value_test_string)

            #match with aged rules
            aged_matching_rule_search_res = search_rules(test_query, filtered_aged_rules)
            aged_matching_rule = aged_matching_rule_search_res['matching_rule']
            aged_activity_value_pred = TemporalEvent('')
            
            if aged_matching_rule:
                aged_activity_value_pred = aged_matching_rule.consequent
                print(f'Matching rule (AGED): {aged_matching_rule.get_rule_string_repr()} {"##SUPPORT:":>20} {aged_matching_rule.support}')
                print(f'Actual sleep value: {activity_value_test}, Predicted sleep value: {aged_activity_value_pred}')
            else:
                print('NO MATCHING RULE FOUND')
            print('CORRECT PREDICTION') if activity_value_test == aged_activity_value_pred else print('WRONG PREDICTION')
            aged_results.append(activity_value_test == aged_activity_value_pred)

            #match with classic rules
            classic_matching_rule_search_res = search_rules(test_query, filtered_classic_rules)
            classic_matching_rule = classic_matching_rule_search_res['matching_rule']
            classic_activity_value_pred = TemporalEvent('')

            if classic_matching_rule:  # can be null when it doesn't find matching rules
                classic_activity_value_pred = classic_matching_rule.consequent
                print(f'Matching rule (CLASS): {classic_matching_rule.get_rule_string_repr()} {"##SUPPORT:":>20} {classic_matching_rule.support}')
                print(f'Actual sleep value: {activity_value_test}, Predicted sleep value: {classic_activity_value_pred}')
            else:
                print('NO MATCHING RULE FOUND')
            print('CORRECT PREDICTION') if activity_value_test == classic_activity_value_pred else print('WRONG PREDICTION')
            print()
            classic_results.append(activity_value_test == classic_activity_value_pred)

            #iterate
            query_range_end += 1
            query_range_start = query_range_end + temporal_window + 1

        aged_accuracy = sum(aged_results) / len(aged_results)
        classic_accuracy = sum(classic_results) / len(classic_results)
        print(f'Analyzed {len(aged_results)} test datapoints.\nAged Apriori accuracy: {aged_accuracy}\nClassic Apriori accuracy: {classic_accuracy}')    
        results.append([aged_accuracy, classic_accuracy, len(train_dataset), len(test_dataset), len(aged_rules), len(classic_rules), aged_time, classic_time, len_dataset, min_support])
    
    results_df = pd.DataFrame(results, index=df_index, columns=['Aged Accuracy','Classic Accuracy', 'Train Size', 'Test Size', 'Aged Rules', 'Classic Rules', 'Aged Time', 'Classic Time', 'Dataset Size', 'Min Support'])
    results_df.to_csv(f"experiments/crop{crop_index}_CONTEXT{context_level}_{df_index[0]}_{activity_type}{activity_value}_TW{temporal_window}_NB{number_of_bins}_MS{min_support}_MC{min_confidence}_df.csv")   

if __name__ == '__main__':
    validate_rules()
