# ACTER framework

### Preparing the dataset

The folder `datasets` contains the two datasets used in this work: PMData and Custom.
In order to run the code with the PMData it is necessary to download the original dataset provided in [V. Thambawita et al., PMData: a sports logging dataset](https://dl.acm.org/doi/10.1145/3339825.3394926). Therefore, it is necessary to copy the files of each user within the corresponding `fitbit` folder.

### Running the code

###### Anomaly detection

There is a Jupyter Notebook in the folder `anomaly`. It runs the code that generates the files in the folder `experiments` necessary to run ALBA and LBA with the contextual information inferred.

###### ALBA - AgedLookBackApriori

```bash
cd aged_apriori
python validation.py
```

It is possible to customize the configurations by modifying the `aged_apriori/config.yaml` file.
In the file, there are the several parameters, following the most relevant ones:

- **`dataset_index`**: Specifies the index of the dataset to be processed ( `0` - PMData,   `1` - Custom).
- **`activity_type_index`**: Defines the index for the specific type of activity to analyze. The value of `-1` means the sleep score.
- **`activity_value`**: Determines the target activity value for filtering or analysis. A value of `-1` indicates no specific value is targeted.
- **`context_level`**: Indicates the level of context to consider in the analysis
  - `0` no context
  - `1` Weekends
  - `2` Weekends and Holidays separated 
  - `4` Holidays 
  - `5` Holidays and Weekends in same column
  - `6` Inferred context (Anomalies)
- **`temporal_window`**: Sets the size of the temporal window for the analysis (default: `3`).
- **`min_support`**: Minimum support threshold for patterns in the data.
- **`min_confidence`**: Minimum confidence level required.
- **`number_of_bins`**: Number of bins used for discretizing continuous variables.
- **`type_fun_discritize`**: Indicates the discretization function type.
- **`thresold_anomaly`**: Threshold value used for the inferred context files, mandatory only when the `context_level` is set to `6`.
- **`time_steps_anomaly`**: Number of time steps considered when evaluating anomalies , mandatory only when the `context_level` is set to `6`.
- **`crop`**: Boolean flag indicating whether to crop or limit the data range for analysis; when set to True, all users have 250 days.

## Reference 📚
If you use our framework in your work, please kindly cite our papers:

```
@article{DALLAVECCHIA2026102666,
  title = {ACTER: Activity Customization through Timely and Explainable Recommendations},
  journal = {Information Systems},
  volume = {138},
  pages = {102666},
  year = {2026},
  issn = {0306-4379},
  doi = {https://doi.org/10.1016/j.is.2025.102666},
  url = {https://www.sciencedirect.com/science/article/pii/S0306437925001528},
  author = {Anna {Dalla Vecchia} and Niccolò Marastoni and Barbara Oliboni and Elisa Quintarelli},
  keywords = {Explainable recommendations, Contextual rules, Context inference},
}

@INPROCEEDINGS{10740439,
  author={Vecchia, Anna Dalla and Marastoni, Niccolò and Quintarelli, Elisa},
  booktitle={2024 IEEE 18th International Conference on Application of Information and Communication Technologies (AICT)}, 
  title={Anomaly detection to infer context changes in temporal data}, 
  year={2024},
  volume={},
  number={},
  pages={1-6},
  keywords={Wearable Health Monitoring Systems;Feature extraction;Information and communication technology;Anomaly detection;Recommender systems;Long short term memory;Meteorology;Context modeling;Anomaly detection;LSTM;contextual features},
  doi={10.1109/AICT61888.2024.10740439}
}

@InProceedings{10.1007/978-3-031-39831-5_7,
  author="Dalla Vecchia, Anna
  and Marastoni, Niccol{\`o}
  and Oliboni, Barbara
  and Quintarelli, Elisa",
  title="The Synergies of Context and Data Aging in Recommendations",
  booktitle="Big Data Analytics and Knowledge Discovery",
  year="2023",
  publisher="Springer Nature Switzerland",
  address="Cham",
  pages="80--87",
  abstract="In this paper, we investigate the synergies of data aging and contextual information in data mining techniques used to infer frequent, up-to-date, and contextual user behaviours that enable making recommendations on actions to take or avoid in order to fulfill a specific positive goal. We conduct experiments in two different domains: wearable devices and smart TVs.",
  isbn="978-3-031-39831-5"
}

@InProceedings{10.1007/978-3-031-42941-5_18,
  author="Dalla Vecchia, Anna
  and Marastoni, Niccol{\`o}
  and Migliorini, Sara
  and Oliboni, Barbara
  and Quintarelli, Elisa",
  title="Mining Totally Ordered Sequential Rules to Provide Timely Recommendations",
  booktitle="New Trends in Database and Information Systems",
  year="2023",
  publisher="Springer Nature Switzerland",
  address="Cham",
  pages="197--207",
  abstract="In this paper we show the importance of mining totally ordered sequential rules, and in particular we propose an extension of sequential rules where not only the antecedent precedes the consequent, but their itemsets are labelled with an explicit representation of their relative order. This allows us to provide more precise timely recommendations. Our technique has been applied to a real-world scenario regarding the provision of tailored suggestions for supermarket shopping activities.",
  isbn="978-3-031-42941-5"
}
```
