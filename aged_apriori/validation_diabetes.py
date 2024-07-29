from typing import Tuple
from numpy import number
import pandas as pd
import time
import yaml
from data_classes import DiabetesDataset, TemporalEvent
from datasets import generate_dataset_diabetes_from_user
from experiments import diabetes_sort_rules, diabetes_search_rules
from utils import call_aged_apriori, call_classic_apriori, clean_context, temporal_from_strings, wrapper_function_diabets
from types import FunctionType
from profilehooks import profile

datasets = DiabetesDataset

def get_auditel_validation_params_decorator(func: FunctionType) -> FunctionType:
    def inner() -> None:
        # Load config from YAML file
        with open("config_diabets.yaml", "r") as fp:
            args = yaml.safe_load(fp)

        print('\nConfig loaded\n')

        context_level = args['context_level']
        temporal_window = args['temporal_window']
        min_support = args['min_support']
        min_confidence = args['min_confidence']
        exponential_decay = args['exponential_decay']
        minutes_event_interval = args['minutes_event_interval']
         
        func(datasets(), context_level, temporal_window,  min_support, min_confidence, exponential_decay, minutes_event_interval)
    
    return inner 

@get_auditel_validation_params_decorator
def validate_rules(dataset_diabetes: DiabetesDataset, context_level:int, temporal_window: int, min_support: float, min_confidence: float, exponential_decay: bool, minutes_event_interval: int) -> None:
    results = []
    df_index = []
    
    for user_index in range(len(dataset_diabetes.users_list)):
        user_name = dataset_diabetes.get_user_name(user_index)
        print(f'Validating experiments for user {user_name}')
      
        dataset = generate_dataset_diabetes_from_user(dataset_diabetes, user_index)

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
        aged_rules = wrapper_function_diabets(train_dataset, apriori_function=call_aged_apriori, temporal_window=temporal_window, 
                                min_support=min_support, min_confidence=min_confidence, exponential_decay=exponential_decay, context_level=context_level)
    
        aged_time = time.time() - aged_time

        
        #generate rules from training dataset with classic apriori
        classic_time = time.time()
        classic_rules = wrapper_function_diabets(train_dataset, apriori_function=call_classic_apriori, temporal_window=temporal_window, 
                                min_support=min_support, min_confidence=min_confidence, exponential_decay=exponential_decay, context_level=context_level)
    
        classic_time = time.time() - classic_time
        

        #print('\n\nAGED RULES\n',len(aged_rules))
        #print(aged_rules)
        print('Rules_mined\n\n')

        fd = open(f'diabetes_rules/discretizationV5_aged_rules_user{user_name}_minutes{minutes_event_interval}_tw{temporal_window}_conf{min_confidence}_sup{min_support}.txt', 'w')
        for r in diabetes_sort_rules(aged_rules):
            fd.write(str(r) + '\n')
        fd.close()

        fd = open(f'diabetes_rules/discretizationV5_classic_rules_user{user_name}_minutes{minutes_event_interval}_tw{temporal_window}_conf{min_confidence}_sup{min_support}.txt', 'w')
        for r in diabetes_sort_rules(classic_rules):
            fd.write(str(r) + '\n')
        fd.close()

        #generate test queries
        fd = open(f'diabetes_rules/QUERYv1_discretizationV5_classic_rules_user{user_name}_minutes{minutes_event_interval}_tw{temporal_window}_conf{min_confidence}_sup{min_support}.txt', 'w')
        
        sliding_window = temporal_window if temporal_window < len(test_dataset) else len(test_dataset)
        query_range_end = 0
        query_range_start = query_range_end + sliding_window + 1
        classic_results = []
        aged_results = []
        number_queires = 0
        while query_range_start - sliding_window + 1 <= test_index:
            if query_range_start > test_index:  
                sliding_window -= 1
            test_query = []
            query_validation = True 
            glucose_test_string = None
            counter = 1
            gap = False

            test_dataset_in_range = test_dataset[-query_range_start:-query_range_end]
            if query_range_end == 0:
                test_dataset_in_range = test_dataset[-query_range_start:]

            for event in test_dataset_in_range[:-1]: 
                if gap:
                    test_query = []
                    gap = False

                if 'G' in event:
                    gap = True
                else:
                    timeloc = sliding_window + 1 - counter 
                    list_of_categories = temporal_from_strings([event], timeloc=timeloc)[0]
                    test_query = test_query + list_of_categories
                counter += 1

            if 'G' in test_dataset_in_range[-1]:
                gap = True
            else:
                test_query = test_query + temporal_from_strings([test_dataset_in_range[-1][0]], timeloc=0)[0] #add last event
                glucose_test_string = test_dataset_in_range[-1][1] #test only glucose level

            if gap:
                query_validation = False
                    
            if query_validation: #True if in the query there are no gaps
                
                output = f'\nQuery: {test_query}'
                fd.write(output+'\n')
                print(output)
                glucose_temporal_event_test = TemporalEvent(glucose_test_string)
                output = f'Actual value: {glucose_temporal_event_test}\n'
                fd.write(output+'\n')
                print(output)
                
                #match with classic rules
                classic_matching_rule_search_res = diabetes_search_rules(test_query, classic_rules) 
                classic_matching_rule = classic_matching_rule_search_res['matching_rule']
                classic_matching_type = classic_matching_rule_search_res['match_type']
                classic_glucose_value_pred = TemporalEvent('')

                if classic_matching_rule:  # can be null when it doesn't find matching rules
                    classic_glucose_value_pred = classic_matching_rule.consequent
                    output = f'Match Type (CLASS): {classic_matching_type}\n'
                    output += f'Matching rule (CLASS): {classic_matching_rule.get_rule_string_repr()} {"##SUPPORT:":>20} {classic_matching_rule.support}\n'
                    output += f'Actual category value: {glucose_temporal_event_test}, Predicted category value: {classic_glucose_value_pred}\n'
                    fd.write(output+'\n')
                    print(output)
                else:
                    output = 'NO MATCHING RULE FOUND'
                    fd.write(output+'\n')
                    print(output)
                if glucose_temporal_event_test == classic_glucose_value_pred:
                    output = 'CORRECT PREDICTION\n'  
                else:
                    output = 'WRONG PREDICTION\n'
                fd.write(output+'\n')
                print(output)
                classic_results.append(glucose_temporal_event_test == classic_glucose_value_pred)

                #match with aged rules
                aged_matching_rule_search_res = diabetes_search_rules(test_query, aged_rules) 
                aged_matching_rule = aged_matching_rule_search_res['matching_rule']
                aged_matching_type = aged_matching_rule_search_res['match_type']
                aged_glucose_value_pred = TemporalEvent('')

                if aged_matching_rule:  # can be null when it doesn't find matching rules
                    aged_glucose_value_pred = aged_matching_rule.consequent
                    output = f'Match Type (AGED): {aged_matching_type}\n'
                    output += f'Matching rule (AGED): {aged_matching_rule.get_rule_string_repr()} {"##SUPPORT:":>20} {aged_matching_rule.support}\n'
                    output += f'Actual category value: {glucose_temporal_event_test}, Predicted category value: {aged_glucose_value_pred}'
                    fd.write(output+'\n')
                    print(output)
                else:
                    output = 'NO MATCHING RULE FOUND'
                    fd.write(output+'\n')
                    print(output)
                if glucose_temporal_event_test == aged_glucose_value_pred:
                    output = 'CORRECT PREDICTION\n'  
                else:
                    output = 'WRONG PREDICTION\n'
                fd.write(output+'\n')
                print(output)
                aged_results.append(glucose_temporal_event_test == aged_glucose_value_pred)

            #iterate
            query_range_end += 1
            query_range_start = query_range_end + sliding_window + 1
            number_queires += 1

        classic_accuracy = sum(classic_results) / len(classic_results)
        aged_accuracy = sum(aged_results) / len(aged_results)
    
        output = f'Analyzed {len(classic_results)} test datapoints.\nAged Apriori accuracy: {aged_accuracy}\nClassic Apriori accuracy: {classic_accuracy}\n'
        output += '*'*30 + '\n'
        fd.write(output+'\n')
        print(output)

    fd.close()      

        

if __name__ == '__main__':
    validate_rules()
