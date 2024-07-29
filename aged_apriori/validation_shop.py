from typing import Tuple
from numpy import number
import pandas as pd
import time
import yaml
from data_classes import TemporalEvent, ShopDataset
from datasets import generate_shop_dataset_from_user
from experiments import shop_search_rules
from utils import call_classic_apriori, temporal_from_strings, wrapper_function_shop
from types import FunctionType
from profilehooks import profile

datasets = ShopDataset

#SHOP DECORATOR
def get_shop_validation_params_decorator(func: FunctionType) -> FunctionType:
    def inner() -> None:
        # Load config from YAML file
        with open("config_shop.yaml", "r") as fp:
            args = yaml.safe_load(fp)

        print('\nConfig loaded\n')

        temporal_window = args['temporal_window']
        min_support = args['min_support']
        min_confidence = args['min_confidence']
        exponential_decay = args['exponential_decay']
         
        func(datasets(), temporal_window,  min_support, min_confidence, exponential_decay)
    
    return inner 

@get_shop_validation_params_decorator
def shop_rules(shop_dataset: ShopDataset, temporal_window: int, min_support: float, min_confidence: float, exponential_decay: bool) -> None:
    results = []
    df_index = []

    for user_index in range(len(shop_dataset.users_list)):
        user_name = shop_dataset.get_user_name(user_index)
        print('\n\n-----------------------------------')
        print(f'Validating experiments for user {user_name}')

        dataset = generate_shop_dataset_from_user(shop_dataset, user_index)
        
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

        #generate rules from training dataset with classic apriori
        classic_time = time.time()
        classic_rules = wrapper_function_shop(train_dataset, apriori_function=call_classic_apriori, temporal_window=temporal_window, 
                                min_support=min_support, min_confidence=min_confidence)
        classic_time = time.time() - classic_time

        #saving to file generated rules
        with open(f"experiments/RULES_SHOP_user{user_name}_TW{temporal_window}_MS{min_support}_MC{min_confidence}_df.csv", 'w') as f:
            f.write('rule,support,confidence\n')
            for rule in classic_rules:
                f.write(rule.get_rule_string_repr() + ',' + str(rule.support) + ',' + str(rule.confidence) + '\n')
        
        #generate test queries
        sliding_window = temporal_window if temporal_window < len(test_dataset) else len(test_dataset)
        query_range_end = 0
        query_range_start = query_range_end + sliding_window + 1
        classic_results = []
        number_queires = 0
        while query_range_start - sliding_window + 1 <= test_index:
            if query_range_start > test_index:  
                sliding_window -= 1
            test_query = []
            query_validation = True 
            category_test_string = None
            counter = 1

            test_dataset_in_range = test_dataset[-query_range_start:-query_range_end]
            if query_range_end == 0:
                test_dataset_in_range = test_dataset[-query_range_start:]

            for category in test_dataset_in_range[:-1]: 
                timeloc = sliding_window + 1 - counter 
                list_of_categories = temporal_from_strings([category], timeloc=timeloc)[0]
                test_query = test_query + list_of_categories
                counter += 1

            category_test_string = test_dataset_in_range[-1]
                    
            if query_validation: #True if in the query there are no gaps
                
                print(f'\nQuery: {test_query}')
                category_value_test = temporal_from_strings([category_test_string], timeloc=0)[0]
                print(f'Actual value: {category_value_test}\n')
                
                #match with classic rules
                classic_matching_rule_search_res = shop_search_rules(test_query, classic_rules)
                #classic_matching_rule_type = classic_matching_rule_search_res[0]['match_type']
                classic_category_value_pred = TemporalEvent('')

                if classic_matching_rule_search_res:  # can be null when it doesn't find matching rules
                    for rule_res in classic_matching_rule_search_res:
                        rule_match = rule_res['matching_rule']
                        classic_category_value_pred = rule_match.consequent
                        print(f'Matching rule:\n {rule_match.get_rule_string_repr()} {"##SUPPORT:":>20} {rule_match.support}')
                        print(f'Predicted value: {classic_category_value_pred}')
                        print('CORRECT PREDICTION') if classic_category_value_pred in category_value_test else print('WRONG PREDICTION')
                        print()
                        classic_results.append(classic_category_value_pred in category_value_test)
                    
                    
                else:
                    print('NO MATCHING RULE FOUND')
                
                print()

            #iterate
            query_range_end += 1
            query_range_start = query_range_end + sliding_window + 1
            number_queires += 1
        
        if len(classic_results) == 0:
            classic_accuracy = 0
        else:
            classic_accuracy = sum(classic_results) / len(classic_results)
        print(f'Analyzed {number_queires} queries.\nFound {len(classic_results)} rule matched.\nClassic Apriori accuracy: {classic_accuracy}')    
        results.append([classic_accuracy, len(train_dataset), len(test_dataset), len(classic_rules), classic_time, len_dataset, min_support])
        
    
    results_df = pd.DataFrame(results, index=df_index, columns=['Classic Accuracy', 'Train Size', 'Test Size', 'Classic Rules', 'Classic Time', 'Dataset Size', 'Min Support'])
    results_df.to_csv(f"experiments/ACCURACIES_SHOP_{df_index[0]}_TW{temporal_window}_MS{min_support}_MC{min_confidence}_df.csv")   
    

if __name__ == '__main__':
    shop_rules()
