from typing import Tuple
from numpy import number
import pandas as pd
import time
import yaml
from data_classes import FitbitDataSet, TemporalEvent, CustomUsersFitbitDataset, PMDataFitbitDataset, AuditelDataset
from datasets import generate_dataset_from_user, generate_context_from_user, generate_auditel_dataset_from_user
from experiments import filter_rules, get_rules_from_user, search_rules, sort_rules, auditel_search_rules
from utils import call_aged_apriori, call_classic_apriori, clean_context, temporal_from_strings, wrapper_function, wrapper_function_auditel
from types import FunctionType
from profilehooks import profile

datasets = [PMDataFitbitDataset, CustomUsersFitbitDataset]
datasets_auditel = AuditelDataset

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

#AUDITEL DECORATOR
def get_auditel_validation_params_decorator(func: FunctionType) -> FunctionType:
    def inner() -> None:
        # Load config from YAML file
        with open("config_auditel.yaml", "r") as fp:
            args = yaml.safe_load(fp)

        print('\nConfig loaded\n')

        context_level = args['context_level']
        temporal_window = args['temporal_window']
        min_support = args['min_support']
        min_confidence = args['min_confidence']
        exponential_decay = args['exponential_decay']
         
        func(datasets_auditel(), context_level, temporal_window,  min_support, min_confidence, exponential_decay)
    
    return inner 

@profile(immediate=True)
@get_validation_params_decorator
def compare_classic_apriori_with_aged_apriori(fitbit_dataset: FitbitDataSet, dataset_index:int, activity_type: str, activity_value: int, context_level:int, temporal_window: int, number_of_bins: int, min_support: float, min_confidence: float) -> None:
    default_skip_users = 0
    skip_users = int(input(f'skip n users (default = {default_skip_users}): ') or default_skip_users)
   
    number_of_rules_per_experiment = 20                 # TODO #8 this value should probably not be hardcoded
    
    for user_index in list(range(0, len(fitbit_dataset.users_list))):     # user p12 doesn't have light activity json
        user_name = fitbit_dataset.get_user_name(user_index)
        print(f'Generating experiments for {user_name}')
        print(f'user_index = {user_index}')
        if user_index < skip_users:
            continue

        classic_rules = get_rules_from_user(fitbit_dataset, user_index, activity_value=activity_value, apriori_function=call_classic_apriori, temporal_window=temporal_window, number_of_bins=number_of_bins, min_support=min_support, min_confidence=min_confidence)
        aged_rules = get_rules_from_user(fitbit_dataset, user_index, activity_value=activity_value, apriori_function=call_aged_apriori, temporal_window=temporal_window, number_of_bins=number_of_bins, min_support=min_support, min_confidence=min_confidence)
    
        top_classic_rules_s = pd.Series([rule.get_rule_string_repr() for rule in classic_rules[:number_of_rules_per_experiment]], name='classic rules')
        top_classic_support_s = pd.Series([rule.support for rule in classic_rules[:number_of_rules_per_experiment]], name='classic support')
        top_classic_confidence_s = pd.Series([rule.confidence for rule in classic_rules[:number_of_rules_per_experiment]], name='classic confidence')
        
        top_aged_rules_s = pd.Series([rule.get_rule_string_repr() for rule in aged_rules[:number_of_rules_per_experiment]], name='aged rules')
        top_aged_support_s = pd.Series([rule.support for rule in aged_rules[:number_of_rules_per_experiment]], name='aged support')
        top_aged_confidence_s = pd.Series([rule.confidence for rule in aged_rules[:number_of_rules_per_experiment]], name='aged confidence')
        
        top_20_df = pd.concat([top_classic_rules_s, top_classic_support_s, top_classic_confidence_s, top_aged_rules_s, top_aged_support_s, top_aged_confidence_s], axis=1)
        top_20_df.to_csv(f"experiments/TOP_RULES_{user_name}_SL{activity_value}_TW{temporal_window}_NB{number_of_bins}_MS{min_support}_MC{min_confidence}_top_{number_of_rules_per_experiment}_df.csv")
        
        print(f'DONE Generating experiments for {user_name}')

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

            test_dataset_in_range = test_dataset[-query_range_start:-query_range_end]
            if query_range_end == 0:
                test_dataset_in_range = test_dataset[-query_range_start:]

            for list_of_strings in test_dataset_in_range:  
                current_day = list_of_strings[:-1]
                timeloc = temporal_window + 1 - counter 
                list_of_activities = temporal_from_strings([current_day], timeloc=timeloc)[0]
                test_query = test_query + list_of_activities
                activity_value_test_string = [x for x in list_of_strings if x.startswith(activity_type)][0]
                counter += 1
            
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
    results_df.to_csv(f"experiments/ACCURACIES_CONTEXT{context_level}_{df_index[0]}_{activity_type}{activity_value}_TW{temporal_window}_NB{number_of_bins}_MS{min_support}_MC{min_confidence}_df.csv")   


@get_auditel_validation_params_decorator
def validate_auditel_rules(auditel_dataset: AuditelDataset, context_level:int, temporal_window: int, min_support: float, min_confidence: float, exponential_decay: bool) -> None:
    results = []
    df_index = []
    
    for user_index in range(len(auditel_dataset.users_list)):
        user_name = auditel_dataset.get_user_name(user_index)
        print(f'Validating experiments for user {user_name}')
      
        dataset = generate_auditel_dataset_from_user(auditel_dataset, user_index, context_level)

        len_dataset = len(dataset)
        print(f'DATASET SIZE: {len_dataset}')
        
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
        aged_rules = wrapper_function_auditel(train_dataset, apriori_function=call_aged_apriori, temporal_window=temporal_window, 
                                min_support=min_support, min_confidence=min_confidence, exponential_decay=exponential_decay, context_level=context_level)
        filtered_aged_rules = aged_rules        #filter_rules(aged_rules) # useless since we are matching with queries that don't have sleep in antecedent?
        aged_time = time.time() - aged_time
        

        #generate rules from training dataset with classic apriori
        classic_time = time.time()
        classic_rules = wrapper_function_auditel(train_dataset, apriori_function=call_classic_apriori, temporal_window=temporal_window, 
                                min_support=min_support, min_confidence=min_confidence, context_level=context_level)
        filtered_classic_rules = classic_rules  #filter_rules(classic_rules)
        classic_time = time.time() - classic_time

        print('\n\nAGED RULES\n',len(aged_rules))
        print('\n\nCLASSIC RULES\n',len(aged_rules), '\n\n')

        #generate test queries
        sliding_window = temporal_window
        query_range_end = 0
        query_range_start = query_range_end + sliding_window + 1
        aged_results = []
        classic_results = []
        while query_range_start - sliding_window + 1 <= test_index:
            if query_range_start > test_index:  
                sliding_window -= 1
            test_query = []
            query_validation = True 
            category_test_string = None
            counter = 1
            gap = False

            test_dataset_in_range = test_dataset[-query_range_start:-query_range_end]
            if query_range_end == 0:
                test_dataset_in_range = test_dataset[-query_range_start:]

            for category in test_dataset_in_range[:-1]: 
                if gap:             #found gap, reset test_query
                    test_query = []
                    gap = False

                if 'G' in category: #check if gap in train value
                    gap = True
                else:
                    timeloc = sliding_window + 1 - counter 
                    list_of_categories = temporal_from_strings([category], timeloc=timeloc)[0]
                    test_query = test_query + list_of_categories
                counter += 1

            category_test_string = test_dataset_in_range[-1][-1]
            if context_level > 0:
                context_test = test_dataset_in_range[-1][:-1]
                test_query = test_query + temporal_from_strings([context_test], timeloc=0)[0]
        
            if gap or category_test_string == 'G' or category_test_string.startswith('$') or category_test_string.startswith('&') or category_test_string.startswith('@') or category_test_string.startswith('*'):
                query_validation = False 

            if query_validation: #True if in the query there are no gaps
                if context_level > 0:
                    test_query = clean_context([test_query])[0] # cleaning the query from repeated contexts
                
                print(f'Query: {test_query}')
                category_value_test = TemporalEvent(category_test_string)
                #print(f'Actual category value: {category_value_test}\n')
                
                #match with aged rules
                aged_matching_rule_search_res = auditel_search_rules(test_query, filtered_aged_rules)
                aged_matching_rule = aged_matching_rule_search_res['matching_rule']
                aged_category_value_pred = TemporalEvent('')

                if aged_matching_rule:
                    aged_category_value_pred = aged_matching_rule.consequent
                    print(f'Matching rule (AGED): {aged_matching_rule.get_rule_string_repr()} {"##SUPPORT:":>20} {aged_matching_rule.support}')
                    print(f'Actual category value: {category_value_test}, Predicted category value: {aged_category_value_pred}')
                else:
                    print('NO MATCHING RULE FOUND')
                print('CORRECT PREDICTION') if category_value_test == aged_category_value_pred else print('WRONG PREDICTION')
                aged_results.append(category_value_test == aged_category_value_pred)
                
                #match with classic rules
                classic_matching_rule_search_res = auditel_search_rules(test_query, filtered_classic_rules)
                classic_matching_rule = classic_matching_rule_search_res['matching_rule']
                classic_category_value_pred = TemporalEvent('')

                if classic_matching_rule:  # can be null when it doesn't find matching rules
                    classic_category_value_pred = classic_matching_rule.consequent
                    print(f'Matching rule (CLASS): {classic_matching_rule.get_rule_string_repr()} {"##SUPPORT:":>20} {classic_matching_rule.support}')
                    print(f'Actual category value: {category_value_test}, Predicted category value: {classic_category_value_pred}')
                else:
                    print('NO MATCHING RULE FOUND')
                print('CORRECT PREDICTION') if category_value_test == classic_category_value_pred else print('WRONG PREDICTION')
                print()
                classic_results.append(category_value_test == classic_category_value_pred)

            #iterate
            query_range_end += 1
            query_range_start = query_range_end + sliding_window + 1

        aged_accuracy = sum(aged_results) / len(aged_results)
        classic_accuracy = sum(classic_results) / len(classic_results)
        print(f'Analyzed {len(aged_results)} test datapoints.\nAged Apriori accuracy: {aged_accuracy}\nClassic Apriori accuracy: {classic_accuracy}')    
        results.append([aged_accuracy, classic_accuracy, len(train_dataset), len(test_dataset), len(aged_rules), len(classic_rules), aged_time, classic_time, len_dataset, min_support])
    
    results_df = pd.DataFrame(results, index=df_index, columns=['Aged Accuracy','Classic Accuracy', 'Train Size', 'Test Size', 'Aged Rules', 'Classic Rules', 'Aged Time', 'Classic Time', 'Dataset Size', 'Min Support'])
    results_df.to_csv(f"experiments/ACCURACIES_AUDITEL_newQuery_CONTEXT{context_level}_{df_index[0]}_TW{temporal_window}_MS{min_support}_MC{min_confidence}_df.csv")   


if __name__ == '__main__':
    #validate_rules()
    validate_auditel_rules()
