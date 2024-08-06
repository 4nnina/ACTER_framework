from __future__ import annotations  # cannot type hint TemporalEvent otherwise
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import os

#PMDATA
main_pmdata_path = '../datasets/pmdata/'
pmdata_users = ['p01', 'p02', 'p03', 'p04', 'p05', 'p06', 'p07', 'p08', 'p09', 'p10', 'p11', 'p13', 'p14', 'p15', 'p16'] # user p12 doesn't have light activity json

#CUSTOM USERS
main_custom_users_path = '../datasets/custom/'
custom_users = ['USER1','USER2', 'USER3', 'USER4']

@dataclass
class TemporalEvent:
    
#A temporal event is identified by a name and a location in time from t0 to t-n.
    name: str
    timeloc: int = 0

    def set_timeloc(self, timeloc: int) -> None:
        self.timeloc = timeloc

    def similar_to(self, other: TemporalEvent) -> bool:
        self_value = int(self.name[-1])
        other_value = int(other.name[-1])
        diff = abs(self_value - other_value)
        return self.name[:2] == other.name[:2] and diff <= 1 and self.timeloc == other.timeloc # TODO #5 this is not great, it's not maintainable at all

    def __repr__(self):
        return self.name + '_t' + str(self.timeloc)

    def __lt__(self, other):
        return self.name < other.name

    def __eq__(self, other):
        return self.name == other.name and self.timeloc == other.timeloc

    def __hash__(self):
        return hash((self.name, self.timeloc))

@dataclass
class Rule:
    antecedent: list[TemporalEvent]
    consequent: TemporalEvent
    confidence: float = 0
    support: float = 0

    def get_rule_string_repr(self) -> str:
        rule_string_repr = ' + '.join([str(x) for x in self.antecedent]) + f'-> {self.consequent}'
        return rule_string_repr.strip()

    def __repr__(self):
        rule_string_repr = self.get_rule_string_repr()
        return f'{rule_string_repr}\nsupport = {self.support}, confidence = {self.confidence}, completeness = {self.get_completeness()}, size = {self.get_size()}'

    def get_size(self):
        return len(self.antecedent) + 1

    def get_completeness(self):                  # counts how many timelocs have items in them
        timelocs = []
        for temporal_event in self.antecedent:
            timeloc = temporal_event.timeloc
            if timeloc not in timelocs:
                timelocs.append(timeloc)

        return len(timelocs)

@dataclass
class DataSet:
    main_data_path: str = ''
    def get_main_data_path(self) -> str:
        return self.main_data_path

    def set_main_data_path(self, new_main_pmdata_path: str) -> None:
        self.main_data_path = new_main_pmdata_path
    
@dataclass
class FitbitDataSet(DataSet):
    users_list:list[str] = field(default_factory=list)
    def get_user_name(self, user_index: int) -> str:
        if user_index < 0 or user_index >= len(self.users_list):
            raise ValueError(f'user_index has to be between 0 and {len(self.users_list) -1}')
        return self.users_list[user_index]

    def get_user_path(self, user_index: int) -> str:
        user = self.get_user_name(user_index)
        data_path = self.main_data_path + f'{user}/fitbit/'

        if not os.path.exists(data_path):
            raise FileNotFoundError(f'Path {data_path} is not valid, use set_main_pmdata_path() to set a proper one')

        return data_path

@dataclass
class CustomUsersFitbitDataset(FitbitDataSet):
    def __init__(self):
        self.main_data_path = main_custom_users_path
        self.users_list = custom_users

@dataclass
class PMDataFitbitDataset(FitbitDataSet):
    def __init__(self):
        self.main_data_path = main_pmdata_path
        self.users_list = pmdata_users
