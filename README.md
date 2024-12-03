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