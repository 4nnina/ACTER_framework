import numpy as np
import pandas as pd
from data_classes import TemporalEvent, Rule
from experiments import get_rules_from_user, print_rules, search_rules, sort_rules
from utils import get_latex_from_rule, temporal_from_strings, call_classic_apriori, call_aged_apriori, wrapper_function
from temporal_mining import augment_by_column, generate_aged_matrix, aged_apriori, order_itemset
from temporal_mining import find_sleep_patterns, find_sleep_rules
from mock_datasets import get_test_data, get_test_dataset, get_test_itemset
from datasets import discretize_values_sleep_and_activity_df, generate_discretized_pmdata_sleep_and_activity_df_from_user, get_dataset_from_user, get_main_pmdata_path, set_main_pmdata_path, get_pmdata_user_path, get_pmdata_user_dataframes
from datasets import timeshift_sleep_quality_dataframe, generate_sleep_and_activity_df
from mlxtend.preprocessing import TransactionEncoder

def test_temporal_events() -> None:
    L = TemporalEvent('Milk')
    B = TemporalEvent('Cookies')
    P = TemporalEvent('Pasta')
    groceries_dataset = [
                    [L, B],
                    [L, B],
                    [L],
                    [L, P],
                    [B, P],
                    [L, B]
    ]
    print(groceries_dataset)

def test_rules() -> None:
    antecedent = get_test_itemset()
    consequent = TemporalEvent('ZL_3_t0')
    confidence = 1
    support = 0.02

    rule = Rule(antecedent, consequent, confidence, support)

    print(rule)

def test_temporal_from_strings() -> None:
    fitbit_data = get_test_data()
    dataset = temporal_from_strings(fitbit_data)
    print(dataset)

def test_call_classic_apriori() -> None:
    dataset = get_test_dataset()
    result = call_classic_apriori(dataset)
    print(result)

def test_augment_by_column() -> None:
    dataset = get_test_dataset()
    temporal_window = 3
    augmented_dataset = augment_by_column(dataset, temporal_window)
    for el in augmented_dataset:
        print(el)

def test_generate_aged_matrix() -> None:
    fake_matrix = [ [False,  True, False,  True, False, False, False, False,  True, False,  True, False],
                [False, False,  True, False,  True, False, False, False,  True, False, True,  True],
                [False, False,  True,  True, False, False, False, False, False, False,  True, False],
                [False, False,  True, False,  True, False, False, False,  True,  True, True, False],
                [False, False, False,  True, False, False, False, False,  True, False,  True, False],
                [ True, False, False, False, False, False, False,  True, False, False, True,  True],
                [False, False,  True, False, False,  True, False, False, False, False, True,  True],
                [False, False,  True, False, False, False, False, False, False, False, True,  True],
                [False, False,  True, False, False,  True,  True, False, False,  True, True, False]]#,
                #[False,  True, False, False,  True, False, False, False,  True, False,  True, False]]
    fake_df = np.array(fake_matrix)
    aged_matrix = generate_aged_matrix(fake_df, 4)
    aged_df = pd.DataFrame(aged_matrix)

    for i in aged_df:
        col = aged_df[i]
        print(f'{col}')
        print(f'support = {col.sum() / col.size}\n\n')

def test_aged_apriori() -> None:
    dataset = get_test_dataset()
    temporal_window = 3
    min_support = 0.5
    te = TransactionEncoder()
    transactions = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(transactions, columns=te.columns_)
    
    result = aged_apriori(df, temporal_window=temporal_window, min_support=min_support, use_colnames=True)
    print(result)

def test_call_aged_apriori() -> None:
    dataset = get_test_dataset()
    temporal_window = 3
    min_support = 0.5
    result = call_aged_apriori(dataset, temporal_window, min_support)
    print(result)

def test_order_itemset() -> None:
    itemset = get_test_itemset()
    result = order_itemset(itemset)
    print(result)

def test_find_sleep_patterns() -> None:
    dataset = get_test_dataset()
    temporal_window = 3
    sleep_value = 2
    min_support = 0.5
    augmented_dataset = augment_by_column(dataset, temporal_window=temporal_window)
    frequent_itemsets = call_classic_apriori(augmented_dataset)
    sleep_frequent_patterns, frequent_itemsets_dict = find_sleep_patterns(frequent_itemsets, sleep_value=sleep_value, min_support=min_support)
    print(sleep_frequent_patterns)
    print(frequent_itemsets_dict)

def test_find_sleep_rules() -> None:
    dataset = get_test_dataset()
    temporal_window = 3
    sleep_value = 2
    min_support = 0.5
    min_confidence = 0.5
    augmented_dataset = augment_by_column(dataset, temporal_window=temporal_window)
    frequent_itemsets = call_classic_apriori(augmented_dataset)
    sleep_frequent_patterns, frequent_itemsets_dict = find_sleep_patterns(frequent_itemsets, sleep_value=sleep_value, min_support=min_support)
    rules = find_sleep_rules(sleep_frequent_patterns, frequent_itemsets_dict, min_confidence=min_confidence)
    for rule in rules:
        print(rule)

def test_wrapper_function() -> None:
    data = get_test_data()
    temporal_window = 3
    sleep_value = 2
    min_support = 0.5
    min_confidence = 0.5
    rules = wrapper_function(data, sleep_value, apriori_function=call_classic_apriori, temporal_window=temporal_window, min_support=min_support, min_confidence=min_confidence)
    for rule in rules:
        print(rule)

def test_set_and_get_main_pmdata_path() -> None:
    original_data_path = get_main_pmdata_path()
    new_data_path = 'mock/data/path'
    set_main_pmdata_path(new_data_path)
    print(f'original = {original_data_path}\nnew = {get_main_pmdata_path()}')
    set_main_pmdata_path(original_data_path)
    print(f'reverted to {get_main_pmdata_path()}')

def test_get_pmdata_user_path() -> None:
    pmd_data_user_path = get_pmdata_user_path(0)    # happy path
    print(f'got {pmd_data_user_path}')
    try:
        get_pmdata_user_path(16)                        # grumpy path
    except Exception as e:
        print(f'got an exception: {e}')

def test_get_pmdata_user_dataframes() -> None:
    user_index = 4
    light_activity_df, moderate_activity_df, heavy_activity_df, rest_df, sleep_quality_df = get_pmdata_user_dataframes(user_index)
    print(light_activity_df.head())

def test_timeshift_sleep_quality_dataframe() -> None:
    user_index = 4
    light_activity_df, moderate_activity_df, heavy_activity_df, rest_df, sleep_quality_df = get_pmdata_user_dataframes(user_index)
    
    print(f'original: {sleep_quality_df.head(2)}')
    sleep_quality_df = timeshift_sleep_quality_dataframe(sleep_quality_df)
    print(f'timeshifted: {sleep_quality_df.head(2)}')

def test_generate_sleep_and_activity_df() -> None:
    user_index = 4
    light_activity_df, moderate_activity_df, heavy_activity_df, rest_df, sleep_quality_df = get_pmdata_user_dataframes(user_index)
    sleep_and_activity_df = generate_sleep_and_activity_df(light_activity_df, moderate_activity_df, heavy_activity_df, rest_df, sleep_quality_df)
    print(sleep_and_activity_df.head(2))

def test_discretize_values_sleep_and_activity_df() -> None:
    user_index = 4
    light_activity_df, moderate_activity_df, heavy_activity_df, rest_df, sleep_quality_df = get_pmdata_user_dataframes(user_index)
    sleep_and_activity_df = generate_sleep_and_activity_df(light_activity_df, moderate_activity_df, heavy_activity_df, rest_df, sleep_quality_df)
    sleep_and_activity_df = discretize_values_sleep_and_activity_df(sleep_and_activity_df)
    print(sleep_and_activity_df.head(2))

def test_generate_discretized_pmdata_sleep_and_activity_df_from_user() -> None:
    user_index = 4
    sleep_and_activity_df = generate_discretized_pmdata_sleep_and_activity_df_from_user(user_index)
    print(sleep_and_activity_df.head(2))

def test_get_dataset_from_user() -> None:
    user_index = 4
    dataset = get_dataset_from_user(user_index)
    print(dataset[:2])

def test_print_rules() -> None:
    data = get_test_data()
    temporal_window = 3
    sleep_value = 2
    min_support = 0.5
    min_confidence = 0.5
    rules = wrapper_function(data, sleep_value, apriori_function=call_classic_apriori, temporal_window=temporal_window, min_support=min_support, min_confidence=min_confidence)
    print_rules(rules)

def test_sort_rules() -> None:
    data = get_test_data()
    temporal_window = 3
    sleep_value = 2
    min_support = 0.4
    min_confidence = 0.4
    rules = wrapper_function(data, sleep_value, apriori_function=call_classic_apriori, temporal_window=temporal_window, min_support=min_support, min_confidence=min_confidence)
    print('unsorted rules:')
    print_rules(rules)
    sorted_rules = sort_rules(rules)
    print('sorted rules:')
    print_rules(sorted_rules)

def test_search_rules():
    data = get_test_data()
    temporal_window = 3
    sleep_value = 2
    min_support = 0.4
    min_confidence = 0.4
    rules = wrapper_function(data, sleep_value, apriori_function=call_classic_apriori, temporal_window=temporal_window, min_support=min_support, min_confidence=min_confidence)
    
    #EXACT MATCH
    test_query_index = int(len(rules) / 2)
    test_query = rules[test_query_index].antecedent
    res_rule = search_rules(test_query, rules)
    print(f'EXACT MATCH\nquery = {test_query}\nres_rule = {res_rule}\n')

    #MATCH
    test_query = [TemporalEvent('LA_1', 2), TemporalEvent('LA_2', 2), TemporalEvent('HA_2', 1), TemporalEvent('ST_3', 1), TemporalEvent('LA_1', 0)]
    res_rule = search_rules(test_query, rules)
    print(f'MATCH\nquery = {test_query}\nres_rule = {res_rule}\n')

    #PARTIAL MATCH
    test_query = [TemporalEvent('LA_2', 2), TemporalEvent('HA_2', 1), TemporalEvent('ST_3', 1), TemporalEvent('LA_1', 0)]
    res_rule = search_rules(test_query, rules)
    print(f'PARTIAL MATCH\nquery = {test_query}\nres_rule = {res_rule}\n')

    #SIMILAR MATCH
    test_query = [TemporalEvent('LA_2', 2), TemporalEvent('HA_1', 1), TemporalEvent('ST_2', 1), TemporalEvent('LA_2', 0)]
    res_rule = search_rules(test_query, rules)
    print(f'SIMILAR MATCH\nquery = {test_query}\nres_rule = {res_rule}\n')

    #NO MATCH
    test_query = [TemporalEvent('BA_2', 2), TemporalEvent('HGA_1', 1), TemporalEvent('SBT_2', 1), TemporalEvent('LAG_2', 0)]
    res_rule = search_rules(test_query, rules)
    print(f'NO MATCH\nquery = {test_query}\nres_rule = {res_rule}\n')

def test_get_latex_from_rule() -> None:
    data = get_test_data()
    temporal_window = 3
    sleep_value = 2
    min_support = 0.4
    min_confidence = 0.4
    rules = wrapper_function(data, sleep_value, apriori_function=call_classic_apriori, temporal_window=temporal_window, min_support=min_support, min_confidence=min_confidence)
    sorted_rules = sort_rules(rules)
    rule_to_print = sorted_rules[0]
    print(get_latex_from_rule(rule_to_print))

def test_get_rules_from_user() -> None:
    user_index = 4
    sleep_value = 3
    temporal_window = 2
    min_support = 0.04
    min_confidence = 0.8
    rules = get_rules_from_user(user_index, sleep_value=sleep_value, apriori_function=call_aged_apriori, temporal_window=temporal_window, min_support=min_support, min_confidence=min_confidence)
    print('\n\nPRINT RULES')
    print_rules(rules)

if __name__ == '__main__':
    test_utils = False
    test_temporal_mining = False
    test_data_classes = False
    test_datasets = False
    test_experiments = True

    if test_utils:
        test_temporal_from_strings()
        test_call_classic_apriori()
        test_call_aged_apriori()
        test_wrapper_function()
        test_get_latex_from_rule()

    if test_temporal_mining:
        test_augment_by_column()
        test_generate_aged_matrix()
        test_aged_apriori()
        test_order_itemset()
        test_find_sleep_patterns()
        test_find_sleep_rules()

    if test_data_classes:    
        test_temporal_events()
        test_rules()

    if test_datasets:
        test_set_and_get_main_pmdata_path()
        test_get_pmdata_user_path()
        test_get_pmdata_user_dataframes()
        test_timeshift_sleep_quality_dataframe()
        test_generate_sleep_and_activity_df()
        test_discretize_values_sleep_and_activity_df()
        test_generate_discretized_pmdata_sleep_and_activity_df_from_user()
        test_get_dataset_from_user()

    if test_experiments:
        test_print_rules()
        test_sort_rules()
        test_search_rules()
        test_get_rules_from_user()
    