from data_classes import FitbitDataSet, Rule, TemporalEvent
from datasets import generate_dataset_from_user
from utils import call_aged_apriori, wrapper_function
import pandas as pd
pd.options.mode.chained_assignment = None

def print_rules(rules: list[Rule], min_support: float = 0, min_confidence: int = 0) -> None:
    filtered_rules = filter_rules(rules, min_support=min_support, min_confidence=min_confidence)
    for i in range(len(filtered_rules)):
        rule = filtered_rules[i]
        print(i, ')', rule)

def filter_rules(rules: list[Rule], min_support: float = 0, min_confidence: int = 0) -> list[Rule]:
    filtered_rules = []
    for rule in rules:
        sleep_not_in_antecedent = True

        for temporal_event in rule.antecedent:
            sleep_not_in_antecedent = sleep_not_in_antecedent and temporal_event.name[:2] != 'ZL'

        if sleep_not_in_antecedent and rule.support >= min_support and rule.confidence >= min_confidence:
            filtered_rules.append(rule)
    return filtered_rules

def rules_sorter(rule: Rule) -> tuple:
    return (rule.confidence, rule.get_completeness(), rule.support, rule.get_size())

def auditel_rules_sorter(rule: Rule) -> tuple:
    return (rule.confidence, rule.support)

def shop_rules_sorter(rule: Rule) -> tuple:
    return (rule.confidence, rule.support, rule.get_size() - 1) #antecedent size

def diabetes_rules_sorter(rule: Rule) -> tuple:
    return (rule.confidence, rule.get_completeness(), rule.support, rule.get_size())

def sort_rules(rules: list[Rule]) -> list[Rule]:
    sorted_rules = sorted(rules, key=rules_sorter, reverse=True)
    return sorted_rules

def auditel_sort_rules(rules: list[Rule]) -> list[Rule]:
    sorted_rules = sorted(rules, key=auditel_rules_sorter, reverse=True)
    return sorted_rules

def shop_sort_rules(rules: list[Rule]) -> list[Rule]:
    sorted_rules = sorted(rules, key=shop_rules_sorter, reverse=True)
    return sorted_rules

def diabetes_sort_rules(rules: list[Rule]) -> list[Rule]:
    sorted_rules = sorted(rules, key=diabetes_rules_sorter, reverse=True)
    return sorted_rules

def compare_antecedents(query_antecedent: list[TemporalEvent], other_antecedent: list[TemporalEvent]):
    if query_antecedent == other_antecedent:    # exact match
        return 0

    query_rule = Rule(query_antecedent, TemporalEvent('FAKE'))
    other_rule = Rule(other_antecedent, TemporalEvent('FAKE'))

    if query_rule.get_completeness() != other_rule.get_completeness():
        return 4    # this means that not every timeloc is covered

    exact_containment = []

    for temporal_event in other_antecedent:    
        temporal_event_in_query = temporal_event in query_antecedent
        exact_containment.append(temporal_event_in_query)

        if not temporal_event_in_query:
            similar_temporal_event_in_query = any([temporal_event.similar_to(x) for x in query_antecedent])
            if not similar_temporal_event_in_query:     # if there aren't any similar, it's not a match at all
                return 4

    if all(exact_containment):  # match -> all contained 
        return 1
    
    if any(exact_containment):  # partial match -> some contained, other similar
        return 2

    return 3                    # similar match -> only similar

def auditel_shop_compare_antecedents(query_antecedent: list[TemporalEvent], other_antecedent: list[TemporalEvent]):
    if sorted(query_antecedent) == sorted(other_antecedent):    # exact match
        return 0

    for temporal_event in other_antecedent:    
        temporal_event_in_query = temporal_event in query_antecedent

        if not temporal_event_in_query: 
            return 2                            #other has something that query doesn't

    return 1                                    #other is a subset of query

def diabetes_compare_antecedents(query_antecedent: list[TemporalEvent], other_antecedent: list[TemporalEvent]):
    #print(query_antecedent, other_antecedent)
    if sorted(query_antecedent) == sorted(other_antecedent):    # exact match
        return 0

    for temporal_event in other_antecedent:    
        temporal_event_in_query = temporal_event in query_antecedent

        if not temporal_event_in_query: 
            return 2                            #other has something that query doesn't

    return 1

def search_rules(query: list[TemporalEvent], rules: list[Rule]) -> dict:
    '''
    rough idea: scroll through the sorted rules, when you encounter an exact match, return it immediately
    otherwise, keep going and collect one match, one partial match and one similar match
    once the list has been exausted, return the best of the above that is not None
    TODO: #6 handle rule not found error
    '''
    match_types = ['Exact Match', 'Match', 'Partial Match', 'Similar Match', 'NO MATCH']
    sorted_rules = sort_rules(rules)
    best_rule_similarity = 4
    best_rule = None

    for rule in sorted_rules:       
        other_antecedent = rule.antecedent
        antecedent_similarity = compare_antecedents(query, other_antecedent)

        if antecedent_similarity == 0:
            return {
                    'match_type' : match_types[antecedent_similarity],
                    'matching_rule': rule
                 }

        elif antecedent_similarity < best_rule_similarity:
            best_rule = rule
            best_rule_similarity = antecedent_similarity

    return {
            'match_type' : match_types[best_rule_similarity],
            'matching_rule': best_rule
        }

def auditel_search_rules(query: list[TemporalEvent], rules: list[Rule]) -> dict:
    '''
    rough idea: scroll through the sorted rules, when you encounter an exact match, return it immediately
    otherwise, keep going and collect one match once the list has been exausted, 
    return the best of the above that is not None
    TODO: #6 handle rule not found error
    '''
    match_types = ['Exact Match', 'Match', 'NO MATCH']
    sorted_rules = auditel_sort_rules(rules)
    best_rule_similarity = 2
    best_rule = None

    for rule in sorted_rules:       
        if '$' in rule.consequent.name or '&' in rule.consequent.name or '@' in rule.consequent.name or '*' in rule.consequent.name: # ignore rules that have a context as consequent
            continue

        other_antecedent = rule.antecedent
        antecedent_similarity = auditel_shop_compare_antecedents(query, other_antecedent)
        
        if antecedent_similarity == 0:
            return {
                    'match_type' : match_types[antecedent_similarity],
                    'matching_rule': rule
                 }

        elif antecedent_similarity < best_rule_similarity:
            best_rule = rule
            best_rule_similarity = antecedent_similarity

    return {
            'match_type' : match_types[best_rule_similarity],
            'matching_rule': best_rule
        }

def shop_search_rules(query: list[TemporalEvent], rules: list[Rule]) -> list[dict]:
    match_types = ['Exact Match', 'Match', 'NO MATCH']
    sorted_rules = shop_sort_rules(rules)
    best_rule_similarity = 2
    len_rule_matched = 0
    matched_rules = []

    for rule in sorted_rules:       
        other_antecedent = rule.antecedent
        antecedent_similarity = auditel_shop_compare_antecedents(query, other_antecedent)
        
        if antecedent_similarity < best_rule_similarity:
            matched_rules = []
            len_rule_matched = len(other_antecedent)
            matched_rules.append({
                    'match_type' : match_types[antecedent_similarity],
                    'matching_rule': rule
                 })
            best_rule_similarity = antecedent_similarity
        elif antecedent_similarity == best_rule_similarity:
            if len(other_antecedent) < len_rule_matched:
                continue
            len_rule_matched = len(other_antecedent)
            matched_rules.append({
                    'match_type' : match_types[antecedent_similarity],
                    'matching_rule': rule
                 })

    return matched_rules


def diabetes_search_rules(query: list[TemporalEvent], rules: list[Rule]) -> dict:
    match_types = ['Exact Match', 'Match', 'NO MATCH']
    sorted_rules = diabetes_sort_rules(rules)
    best_rule_similarity = 2
    best_rule = None

    for rule in sorted_rules:       
        other_antecedent = rule.antecedent
        antecedent_similarity = diabetes_compare_antecedents(query, other_antecedent)

        if antecedent_similarity == 0:
            return {
                    'match_type' : match_types[antecedent_similarity],
                    'matching_rule': rule
                 }

        elif antecedent_similarity < best_rule_similarity:
            best_rule = rule
            best_rule_similarity = antecedent_similarity

    return {
            'match_type' : match_types[best_rule_similarity],
            'matching_rule': best_rule
        }

def get_rules_from_user(fitbit_dataset: FitbitDataSet, user_index: int, sleep_value: int = 3, apriori_function=call_aged_apriori, temporal_window: int = 2, number_of_bins: int = 3, min_support: float = 0.2 ,min_confidence: float = 1.0) -> list[Rule]:
    dataset = generate_dataset_from_user(fitbit_dataset, user_index, number_of_bins)
    rules = wrapper_function(dataset, sleep_value, apriori_function=apriori_function, temporal_window=temporal_window, min_support=min_support, min_confidence=min_confidence)
    filtered_rules = filter_rules(rules)
    sorted_rules = sort_rules(filtered_rules)
    return sorted_rules
