from data_classes import TemporalEvent, Rule
import numpy as np
import pandas as pd
from copy import copy

def augment_by_column(dataset: list[list[TemporalEvent]], temporal_window: int) -> list[list[TemporalEvent]]:
    # it starts by populating the last column (t0) and then it goes backwards.
    if temporal_window < 0:
        raise ValueError('Cannot have a null or negative temporal window')

    augmented_dataset = []
    for i in range(len(dataset)):
        augmented_dataset.append(dataset[i])

    for j in range(temporal_window):
        for i in range(j+1, len(dataset)):
            augmented_row = []
            for el in dataset[i-(1+j)]:
                el_copy = copy(el) 
                el_copy.set_timeloc(j+1)
                augmented_row.append(el_copy)

            augmented_dataset[i] = augmented_row + augmented_dataset[i]

    return augmented_dataset

def generate_exponentially_aged_matrix(x, temporal_window, decay_rate=0.4): # TODO #18 either propagate decay_rate to previous methods or move it into the body
    matrix_height = x.shape[0]
    x = x*1
    f_t = []
    aged_total = 0
    
    for i in np.arange(len(x), 0, -1):
        exponent = -float(i*decay_rate)
        aging_factor = 2**exponent
        aged_total += aging_factor
        f_t.append(aging_factor)

    aged_matrix = (x.T * np.array(f_t)).T
    matrix_height = aged_matrix.shape[0]
    make_up = matrix_height - aged_total                     
    make_up_factor = make_up / matrix_height                    # this assumes equally distributed makeup
    make_up_matrix =  x * make_up_factor

    aged_matrix = np.array(aged_matrix)
    aged_matrix = aged_matrix + make_up_matrix

    return aged_matrix


def generate_aged_matrix(x: np.ndarray, temporal_window:int) -> np.ndarray:
    matrix_height = x.shape[0]

    if temporal_window < 1:
        raise ValueError('Temporal window has to be at least 1')
    elif temporal_window > matrix_height:
        raise ValueError('Temporal window cannot be bigger than the number of rows')

    x = x * 1

    matrix_to_age = matrix_height - temporal_window             # portion of the matrix that needs to be aged
    aging_factor =  1 / (matrix_to_age + 1)                     # +1 otherwise the last element of the aged matrix will be the same as the first element of the non-aged
    
    aged_matrix = []
    penalty = 0
    for row in x[:-temporal_window]:                            # all the rows that need to be aged
        penalty += aging_factor
        penalized_row = row * penalty
        aged_matrix.append(penalized_row)

    for row in x[-temporal_window:]:                            # rows in the temporal window are not aged
        aged_matrix.append(row)

    aged_total = temporal_window + (matrix_to_age / 2)          # this is a simple expansion: n(n+1)/2 * 1/(n+1))
    make_up = matrix_height - aged_total                   
    make_up_factor = make_up / matrix_height                    # this assumes equally distributed makeup
    make_up_matrix = x * make_up_factor

    aged_matrix = np.array(aged_matrix)
    aged_matrix = aged_matrix + make_up_matrix
    return aged_matrix


def generate_new_combinations(old_combinations):
    """
    code stolen from mlxtend:
    http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori

    """
    items_types_in_previous_step = np.unique(old_combinations.flatten())
    for old_combination in old_combinations:
        max_combination = old_combination[-1]
        mask = items_types_in_previous_step > max_combination
        valid_items = items_types_in_previous_step[mask]
        old_tuple = tuple(old_combination)
        for item in valid_items:
            yield from old_tuple
            yield item

def aged_apriori(df, temporal_window=3, min_support=0.5, use_colnames=False, max_len=None, verbose=0, low_memory=False, exponential_decay=False):
    """
    most of this is also stolen from mlxtend, only the matrix is aged 
    and sparse matrices are not supported anymore
    """
    def _support(_x, _n_rows, _is_sparse, exponential_decay):
        aged_matrix_generation_function = generate_aged_matrix
        if exponential_decay:
            aged_matrix_generation_function = generate_exponentially_aged_matrix
        _x = aged_matrix_generation_function(_x, temporal_window=temporal_window)  #TODO #1 investigate if it's correct to age the matrix at each iteration, it's possible that in all iterations after the first there's no link between line order and time
        out = np.sum(_x, axis=0) / _n_rows
        result = np.array(out).reshape(-1)
        #print(f'\n_support\n_x_:\n  {_x}\nresult -> {result}')
        return result

    if min_support <= 0.0:
        raise ValueError(
            "`min_support` must be a positive "
            "number within the interval `(0, 1]`. "
            "Got %s." % min_support
        )


    # dense DataFrame
    X = df.values
    is_sparse = False

    support = _support(X, X.shape[0], is_sparse, exponential_decay)
    ary_col_idx = np.arange(X.shape[1])
    support_dict = {1: support[support >= min_support]}
    itemset_dict = {1: ary_col_idx[support >= min_support].reshape(-1, 1)}
    max_itemset = 1
    rows_count = float(X.shape[0])

    all_ones = np.ones((int(rows_count), 1))

    while max_itemset and max_itemset < (max_len or float("inf")):
        next_max_itemset = max_itemset + 1

        combin = generate_new_combinations(itemset_dict[max_itemset])
        combin = np.fromiter(combin, dtype=int)
        combin = combin.reshape(-1, next_max_itemset)

        if combin.size == 0:
            break
        if verbose:
            print(
                "\rProcessing %d combinations | Sampling itemset size %d"
                % (combin.size, next_max_itemset),
                end="",
            )

     
        _bools = np.all(X[:, combin], axis=2) 
   

        support = _support(np.array(_bools), rows_count, is_sparse, exponential_decay)
        _mask = (support >= min_support).reshape(-1)
        if any(_mask):
            itemset_dict[next_max_itemset] = np.array(combin[_mask])
            support_dict[next_max_itemset] = np.array(support[_mask])
            max_itemset = next_max_itemset
        else:
            # Exit condition
            break

    all_res = []
    for k in sorted(itemset_dict):
        support = pd.Series(support_dict[k])
        itemsets = pd.Series([frozenset(i) for i in itemset_dict[k]], dtype="object")

        res = pd.concat((support, itemsets), axis=1)
        all_res.append(res)

    res_df = pd.concat(all_res)
    res_df.columns = ["support", "itemsets"]
    if use_colnames:
        mapping = {idx: item for idx, item in enumerate(df.columns)}
        res_df["itemsets"] = res_df["itemsets"].apply(
            lambda x: frozenset([mapping[i] for i in x])
        )
    res_df = res_df.reset_index(drop=True)

    if verbose:
        print()  # adds newline if verbose counter was used

    return res_df

def order_itemset(itemset: list[TemporalEvent], activity_type: str) -> list[TemporalEvent]:
    renamed_itemset = []
    for item in itemset:
        if item.name.startswith(activity_type):
            new_item = TemporalEvent('ZZ_' + item.name.split('_')[1], item.timeloc)
            renamed_itemset.append(new_item)
        else:
            renamed_itemset.append(item)
    sorted_itemset = sorted(renamed_itemset, key=lambda x: (-x.timeloc, x.name))

    final_itemset = []
    for item in sorted_itemset:
        if item.name.startswith('ZZ_'):
            new_item = TemporalEvent(activity_type + item.name.split('_')[1], item.timeloc)
            final_itemset.append(new_item)
        else:
            final_itemset.append(item)
    
    return final_itemset

def find_sleep_patterns(frequent_itemsets: pd.DataFrame, activity_type: str = 'ZL_', activity_value:int = 3, min_support: float = 0, min_confidence: float = 0) -> tuple[list[tuple[float, list[TemporalEvent]]], dict]:
    """
    activity_value -> when equal to or less then -1, it returns frequent patterns and itemsets with ALL possible activity values
    Returns:
    frequent_pattern_list:  list of tuples of ordered itemsets (and their supports) that satisfy the filters, 
                            needed to generate rules
    frequent_itemsets_dict: dictionary of frequent itemsets and their support, 
                            needed to retrieve the support of antecedents (which might not satisfy the filters) 
                            and calculate confidence
    """
    frequent_patterns_list = []
    frequent_itemsets_dict = {}
    for index, row in frequent_itemsets.iterrows():
        itemset = row['itemsets']
        support = row['support']
        ordered_itemset = order_itemset(itemset, activity_type)

        key = ordered_itemset[0].__repr__()
        if len(ordered_itemset) > 1:
            key = ' + '.join([x.__repr__() for x in ordered_itemset])
        frequent_itemsets_dict[key] = support

        last_item = ordered_itemset[-1] 
		
        if activity_value > -1: # filter only patterns that have this specific activity value as their last temporal event
            if len(ordered_itemset) > 1 and last_item.name == activity_type + str(activity_value) and last_item.timeloc == 0 and support >= min_support: 
                frequent_patterns_list.append((support, ordered_itemset))

        else: # filter any sleep value
            if len(ordered_itemset) > 1 and last_item.name.startswith(activity_type) and last_item.timeloc == 0 and support >= min_support: 
                frequent_patterns_list.append((support, ordered_itemset))

    return frequent_patterns_list, frequent_itemsets_dict

# TODO #16 #15 min_support is not used in this method
def find_sleep_rules(frequent_patterns_list: list[tuple[float, list[TemporalEvent]]], frequent_itemsets_dict: dict, min_support: float = 0, min_confidence: float = 0) -> list[Rule]: 
    rules = []

    for support, frequent_pattern in frequent_patterns_list:
        antecedents = [x for x in frequent_pattern[:-1]]
        consequent = frequent_pattern[-1]

        antecedents_key = ' + '.join([x.__repr__() for x in antecedents])
        try:
            antecedents_support = frequent_itemsets_dict[antecedents_key.strip()]   # TODO #4 very hacky, should find the root of the problem. It happens with confidence = 0 and support = 0.2 in the test
        except:
            antecedents_support = 0
        
        confidence = 0
        if antecedents_support != 0:
            confidence = support/antecedents_support

        rule = Rule(antecedents, consequent, confidence, support)
        
        if confidence >= min_confidence:
            rules.append(rule)

    return rules

#AUDITEL

def find_auditel_patterns(frequent_itemsets: pd.DataFrame, min_support: float = 0) -> tuple[list[tuple[float, list[TemporalEvent]]], dict]:
    """
    Returns:
    frequent_pattern_list:  list of tuples of itemsets (and their supports) that satisfy the filters, 
                            needed to generate rules
    frequent_itemsets_dict: dictionary of frequent itemsets and their support, 
                            needed to retrieve the support of antecedents (which might not satisfy the filters) 
                            and calculate confidence
    """
    frequent_patterns_list = []
    frequent_itemsets_dict = {}
    for index, row in frequent_itemsets.iterrows():
        itemset = list(row['itemsets'])
        support = row['support']
        ordered_itemset = order_itemset(itemset, '__') #no category starts with __
        
        key = ordered_itemset[0].__repr__()
        if len(ordered_itemset) > 1:
            key = ' + '.join([x.__repr__() for x in ordered_itemset])
        frequent_itemsets_dict[key] = support

        last_item = ordered_itemset[-1]

        if len(ordered_itemset) > 1 and last_item.timeloc == 0 and support >= min_support: 
                frequent_patterns_list.append((support, ordered_itemset))

    return frequent_patterns_list, frequent_itemsets_dict
                
def find_auditel_rules(frequent_patterns_list: list[tuple[float, list[TemporalEvent]]], frequent_itemsets_dict: dict, min_confidence: float = 0) -> list[Rule]: 
    rules = []
    
    for support, frequent_pattern in frequent_patterns_list:
        antecedents = [x for x in frequent_pattern[:-1]]
        consequent = frequent_pattern[-1]

        antecedents_key = ' + '.join([x.__repr__() for x in antecedents])
        try:
            antecedents_support = frequent_itemsets_dict[antecedents_key.strip()]   # TODO #4 very hacky, should find the root of the problem. It happens with confidence = 0 and support = 0.2 in the test
        except:
            antecedents_support = 0
        
        confidence = 0
        if antecedents_support != 0:
            confidence = support/antecedents_support

        rule = Rule(antecedents, consequent, confidence, support)
        
        if confidence >= min_confidence:
            rules.append(rule)

    return rules

#SHOP

def find_shop_patterns(frequent_itemsets: pd.DataFrame, min_support: float = 0) -> tuple[list[tuple[float, list[TemporalEvent]]], dict]:
    """
    Returns:
    frequent_pattern_list:  list of tuples of itemsets (and their supports) that satisfy the filters, 
                            needed to generate rules
    frequent_itemsets_dict: dictionary of frequent itemsets and their support, 
                            needed to retrieve the support of antecedents (which might not satisfy the filters) 
                            and calculate confidence
    """
    frequent_patterns_list = []
    frequent_itemsets_dict = {}
    for index, row in frequent_itemsets.iterrows():
        itemset = list(row['itemsets'])
        support = row['support']
        ordered_itemset = order_itemset(itemset, '__') #no category starts with __
        
        key = ordered_itemset[0].__repr__()
        if len(ordered_itemset) > 1:
            key = ' + '.join([x.__repr__() for x in ordered_itemset])
        frequent_itemsets_dict[key] = support

        last_item = ordered_itemset[-1]

        if len(ordered_itemset) > 1:
            before_last_item = ordered_itemset[-2]

            if last_item.timeloc == 0 and support >= min_support and before_last_item.timeloc != 0: 
                    frequent_patterns_list.append((support, ordered_itemset))

    return frequent_patterns_list, frequent_itemsets_dict
                
def find_shop_rules(frequent_patterns_list: list[tuple[float, list[TemporalEvent]]], frequent_itemsets_dict: dict, min_confidence: float = 0) -> list[Rule]: 
    rules = []
    
    for support, frequent_pattern in frequent_patterns_list:
        antecedents = [x for x in frequent_pattern[:-1]]
        consequent = frequent_pattern[-1]

        antecedents_key = ' + '.join([x.__repr__() for x in antecedents])
        try:
            antecedents_support = frequent_itemsets_dict[antecedents_key.strip()]
        except:
            antecedents_support = 0
        
        confidence = 0
        if antecedents_support != 0:
            confidence = support/antecedents_support

        rule = Rule(antecedents, consequent, confidence, support)
        
        if confidence >= min_confidence:
            rules.append(rule)

    return rules

#DIABETES

def find_diabetes_patterns(frequent_itemsets: pd.DataFrame, min_support: float = 0) -> tuple[list[tuple[float, list[TemporalEvent]]], dict]:
    """
    Returns:
    frequent_pattern_list:  list of tuples of itemsets (and their supports) that satisfy the filters, 
                            needed to generate rules
    frequent_itemsets_dict: dictionary of frequent itemsets and their support, 
                            needed to retrieve the support of antecedents (which might not satisfy the filters) 
                            and calculate confidence
    """
    frequent_patterns_list = []
    frequent_itemsets_dict = {}
    for index, row in frequent_itemsets.iterrows():
        itemset = list(row['itemsets'])
        support = row['support']
        ordered_itemset = order_itemset(itemset, '__') #no category starts with __
        
        key = ordered_itemset[0].__repr__()
        if len(ordered_itemset) > 1:
            key = ' + '.join([x.__repr__() for x in ordered_itemset])
        frequent_itemsets_dict[key] = support

        last_item = ordered_itemset[-1]

        if len(ordered_itemset) > 1 and last_item.timeloc == 0 and support >= min_support: 
                frequent_patterns_list.append((support, ordered_itemset))

    return frequent_patterns_list, frequent_itemsets_dict
                
def find_diabetes_rules(frequent_patterns_list: list[tuple[float, list[TemporalEvent]]], frequent_itemsets_dict: dict, min_confidence: float = 0) -> list[Rule]: 
    rules = []
    
    for support, frequent_pattern in frequent_patterns_list:
        antecedents = []
        consequents = []

        for x in frequent_pattern:
            if x.timeloc == 0:
                consequents.append(x)
            else:
                antecedents.append(x)

        for consequent in consequents:
            antecedents_key = ' + '.join([x.__repr__() for x in antecedents])
            try:
                antecedents_support = frequent_itemsets_dict[antecedents_key.strip()]   # TODO #4 very hacky, should find the root of the problem. It happens with confidence = 0 and support = 0.2 in the test
            except:
                antecedents_support = 0
            
            confidence = 0
            if antecedents_support != 0:
                confidence = support/antecedents_support

            rule = Rule(antecedents, consequent, confidence, support)
            
            if confidence >= min_confidence:
                rules.append(rule)

            if len(consequents) > 1:
                #add in the antecedent other consequents
                extend_antecedent = antecedents.copy()
                for c in consequents:
                    if c != consequent:
                        extend_antecedent.append(c)

                antecedents_key = ' + '.join([x.__repr__() for x in extend_antecedent])
                try:
                    antecedents_support = frequent_itemsets_dict[antecedents_key.strip()]   # TODO #4 very hacky, should find the root of the problem. It happens with confidence = 0 and support = 0.2 in the test
                except:
                    antecedents_support = 0
                
                confidence = 0
                if antecedents_support != 0:
                    confidence = support/antecedents_support

                rule = Rule(extend_antecedent, consequent, confidence, support)
                
                if confidence >= min_confidence:
                    rules.append(rule)
                

    return rules