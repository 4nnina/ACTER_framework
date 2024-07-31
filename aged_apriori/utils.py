from data_classes import TemporalEvent, Rule
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
from types import FunctionType
import pandas as pd
from temporal_mining import aged_apriori, augment_by_column, find_sleep_patterns, find_sleep_rules
import warnings
import functools

def remove_gaps(augmented_dataset : list[list[TemporalEvent]]) -> list[list[TemporalEvent]]:
    clean_dataset = []
    for row in augmented_dataset:
        skip_row = False
        for item in row:
            if item.name == 'G':
                skip_row = True
        if not skip_row:
            clean_dataset.append(row)
    
    return clean_dataset

def temporal_from_strings(dataset: list[list[str]], timeloc: int = 0) -> list[list[TemporalEvent]]:
    new_dataset = []
    for i in range(len(dataset)):
        new_dataset.append([])
        for j in range(len(dataset[i])):
            if dataset[i][j] != '':
                new_dataset[i].append(TemporalEvent(dataset[i][j], timeloc))

    return new_dataset

def call_classic_apriori(dataset: list[list[TemporalEvent]], min_support: float = 0.5) -> pd.DataFrame:
    te = TransactionEncoder()
    transactions = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(transactions, columns=te.columns_)
    return apriori(df, min_support=min_support, use_colnames=True)

def call_aged_apriori(dataset: list[list[TemporalEvent]], temporal_window: int = 3, min_support: float = 0.5, exponential_decay: bool = False) -> pd.DataFrame:
    te = TransactionEncoder()
    transactions = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(transactions, columns=te.columns_)
    return aged_apriori(df, temporal_window=temporal_window, min_support=min_support, use_colnames=True, exponential_decay=exponential_decay)
    
def wrapper_function(dataset: list[list[TemporalEvent]], activity_type: str = 'ZL_', activity_value: int = 3, apriori_function: FunctionType = call_classic_apriori, 
    temporal_window: int = 0, min_support: float = 0.2, min_confidence: float = 0.5, exponential_decay: bool = False, decay_rate=0.4) -> list[Rule]:
    """
    this function allows experiments with both classic and aged apriori
    """
    temporal_dataset = temporal_from_strings(dataset)
    augmented_dataset = augment_by_column(temporal_dataset, temporal_window=temporal_window)
    if exponential_decay:
        frequent_itemsets = apriori_function(augmented_dataset, min_support=min_support, temporal_window=temporal_window, exponential_decay=exponential_decay)
    else:
        frequent_itemsets = apriori_function(augmented_dataset, min_support=min_support)
    sleep_frequent_patterns, frequent_itemsets_dict = find_sleep_patterns(frequent_itemsets, activity_type=activity_type, activity_value=activity_value, min_support=min_support)
    rules = find_sleep_rules(sleep_frequent_patterns, frequent_itemsets_dict, min_confidence=min_confidence)
    return rules

def clean_repetitive_context(row : list[TemporalEvent]) -> list[TemporalEvent]:
    for context_char in ['$','&', '@', '*']:
        context_event = TemporalEvent('')
        new_row = []

        for temporal_event in row:
            if temporal_event.name.startswith(context_char) and (context_event.name != temporal_event.name):  #family context change
                if context_event.name != '':
                    new_row.append(TemporalEvent(context_event.name, temporal_event.timeloc + 1))

                context_event = temporal_event

            elif not temporal_event.name.startswith(context_char) :
                new_row.append(temporal_event)
        if context_event.name != '':
            new_row.append(TemporalEvent(context_event.name, 0))

        row = new_row

    return new_row

def clean_context(augmented_dataset : list[list[TemporalEvent]]) -> list[list[TemporalEvent]]:
    new_augmented_dataset = []

    for row in augmented_dataset:
        new_augmented_dataset.append((clean_repetitive_context(row)))

    return new_augmented_dataset

def get_latex_activity(consequent: TemporalEvent) -> str:
    activity, intensity = consequent.name.split('_')[0], consequent.name.split('_')[1]
    if activity == 'ZL':
        activity = 'SL'
    return activity  + ':' + intensity
    

def get_latex_from_rule(rule: Rule) -> str:
    antecedents, consequent = rule.antecedent, rule.consequent
    latex_rule = r'$\{'
    timeloc = antecedents[0].timeloc
    for i in range(len(antecedents)):
        antecedent = antecedents[i]
        new_timeloc = antecedent.timeloc
        if new_timeloc != timeloc:
            latex_rule += r'\}_{' + str(-int(timeloc)) + r'} \wedge \{'
            timeloc = new_timeloc

        elif i != 0:
            latex_rule += ', '

        latex_rule += get_latex_activity(antecedent) 

    latex_rule += r'\}_{' + str(-int(timeloc)) + '}'
    latex_rule = latex_rule + r' \rightarrow \{' + get_latex_activity(consequent) + r'\}_0$' 
    return latex_rule

def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func