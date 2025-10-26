import numpy as np
import pandas as pd
import os
import sys
import tqdm
import logging
import warnings
from scipy.stats import mannwhitneyu, fisher_exact, chi2_contingency, ttest_ind, kstest, chi2, ks_2samp
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ex
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class StatisticalTests:
    def __init__(self, significance_level):
        self.significance_level = significance_level

    def perform_chi_squared_test(self, contingency_table):
        statistic, p_value, _, _ = chi2_contingency(contingency_table)
        return p_value > self.significance_level, statistic
    
    def perform_t_test(self, data1, data2):
        # perform two-sample t-test on the given data
        # statistic, p_value = ttest_ind(data1, data2, equal_var=False)
        statistic, p_value = ttest_ind(data1, data2)
        return p_value > self.significance_level, statistic

    def perform_ks_test(self, data1, data2):
        # perform two-sample ks-test on the given numerical data
        # statistic, p_value = kstest(data1, data2)
        statistic, p_value = ks_2samp(data1, data2)
        return p_value > self.significance_level, statistic

    def perform_mannwhitneyu_test(self, data1, data2):
        # perform Mann-Whitney U test on the given data
        statistic, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
        return p_value > self.significance_level, statistic


class StatAnalysis:
    def __init__(self, real_table, generated_table, config, significance_level, plot=False):
        self.real_table = real_table
        self.generated_table = generated_table
        self.config = config
        self.significance_level = significance_level
        self.plot = plot
        # self.columns_types = ['numeric', 'numeric_ks', 'categorical_chi2', 'categorical_g', 'null_columns']
        self.columns_types = ['numeric', 'numeric_ks', 'categorical_chi2', 'null_columns']
        
        self.column_statistics = {col_type: {} for col_type in self.columns_types}
        self.accepted_columns = {col_type: [] for col_type in self.columns_types}
        self.aki_accepted_columns = {aki_group: {col_type: [] for col_type in self.columns_types} for aki_group in range(2)}

        self.categorical_columns = self.config['column_order'][:self.config['nc']]
        self.stat_tests = StatisticalTests(significance_level)

    def perform_statistical_tests(self, column, real_data, generated_data, categorical, aki=None):
        test_func_dict = {
            'numeric': [
                (self.stat_tests.perform_t_test, 'numeric'),
                (self.stat_tests.perform_ks_test, 'numeric_ks'),
                ],
            'categorical': [
                (self.stat_tests.perform_chi_squared_test, 'categorical_chi2'),
                ]
            }

        plot_func_dict = {'numeric': self.plot_numerical, 'categorical': self.plot_categorical}
        key_type = 'categorical' if categorical else 'numeric'

        for test_func, key in test_func_dict[key_type]:
            if key_type == 'categorical':
                contingency_table = pd.concat([real_data, generated_data], axis=1).apply(pd.Series.value_counts).fillna(0)
                same_dist, statistic = test_func(contingency_table)
            else:
                same_dist, statistic = test_func(real_data, generated_data)
            if same_dist:
                if aki is None:
                    self.accepted_columns[key].append(column)
                else:
                    self.aki_accepted_columns[aki][key].append(column)

            
            self.column_statistics[key][column] = statistic

        if self.plot:
            plot_func_dict[key_type](column, real_data, generated_data, same_dist)
        

    def analyze(self, count=True, avg=False):
        categorical_columns = self.config['column_order'][:self.config['nc']]
        
        for column in tqdm.tqdm(self.generated_table.columns):
            real_data = self.real_table[column].dropna()
            generated_data = self.generated_table[column].dropna()

            if generated_data.empty:
                self.accepted_columns['null_columns'].append(column)
                continue

            categorical = column in categorical_columns
            self.perform_statistical_tests(column, real_data, generated_data, categorical)

        self.print_results(self.accepted_columns)
        if count:
            return {k: len(v) for k, v in self.accepted_columns.items()}
        elif (count == False) and (avg== False):
            self.column_statistics["null_columns"] = len(self.accepted_columns["null_columns"])
            return self.column_statistics
        else:
            new_dict = {}
            for key in self.column_statistics:
                if key == "null_columns":
                    new_dict[key] = len(self.accepted_columns[key])
                else:
                    data = self.column_statistics[key]
                    if key == "numeric":
                        key += "_t"
                    new_dict[key] = sum([abs(value) for value in data.values()]) / len(data)
            return new_dict
            

    
    def compare_aki_group(self):
        # Load class2raw from config input_path
        input_path = self.config.get('input_path')
        if input_path is None:
            raise ValueError("input_path must be set in config")
        class2raw = pd.read_pickle(os.path.join(input_path, "class2raw.pickle"))
        
        target_columns = self.config['column_order']
        categorical_columns = [i for i in target_columns if i in self.config['column_order'][:self.config['nc']]]

        real_tables = [
            self.real_table[self.real_table['O_AKI'].map(class2raw) == i] 
            for i in range(2)
        ]
        gen_tables = [
            self.generated_table[self.generated_table['O_AKI'].map(class2raw) == i] 
            for i in range(2)
        ]

        for column in tqdm.tqdm(target_columns):
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))

            for idx, (real_table, gen_table) in enumerate(zip(real_tables, gen_tables)):
                real_data = real_table[column].dropna()
                categorical = column in categorical_columns
                generated_data = gen_table[column].dropna()

                # if not categorical: #NOTE: We use min-max clip to enhance results. 
                #     generated_data = generated_data.clip(real_data.min(), real_data.max())

                if generated_data.empty:
                    self.accepted_columns['null_columns'].append(column)
                    continue

                self.perform_statistical_tests(column, real_data, generated_data, categorical, aki=idx)
                
                all_data = pd.concat([real_data, generated_data])
                x_min, x_max = all_data.min(), all_data.max()

                axs[idx].hist(real_data, bins=30, alpha=0.5, label='Real', range=(x_min, x_max))
                axs[idx].hist(generated_data, bins=30, alpha=0.5, label='Generated', range=(x_min, x_max))

                title = f'{column} (AKI={idx})'
                if not categorical:
                    stats = {
                        'Real': (real_data.mean(), real_data.std(), real_data.median()),
                        'Gen': (generated_data.mean(), generated_data.std(), generated_data.median()),
                    }
                    title += '\n' + '\n'.join(
                        f'{k} Median: {m:.2f}, Mean: {me:.2f}, Std: {s:.2f}' 
                        for k, (me, s, m) in stats.items()
                    )

                axs[idx].set_title(title)
                axs[idx].legend(loc='upper right')
                if not categorical:
                    axs[idx].set_xlim([x_min, x_max])
                    axs[idx].set_xticks(np.linspace(x_min, x_max, num=10)) 

            plt.tight_layout()
            plt.savefig(f'figure/aki/{column}_AKI.png')
            plt.close(fig)

        print("AKI=0")
        self.print_results(self.aki_accepted_columns[0])
        print("AKI=1")
        self.print_results(self.aki_accepted_columns[1])

        return



    def print_results(self, accepted_columns):
        print(" ** Statistical test ** ")
        print(f"Same dist. by chi2-test: {len(accepted_columns['categorical_chi2'])} / {self.config['nc']}")
        print(f"Same dist. by T-test: {len(accepted_columns['numeric'])} / {self.config['nn']}")
        print(f"Null columns predicted: {len(accepted_columns['null_columns'])} / {(self.config['nc']+self.config['nn'])}")
        print(f"Null columns predicted: {accepted_columns['null_columns']}")

    def plot_categorical(self, column, real_data, generated_data, same_dist):
        title_text = f"{column} same dist={same_dist}"
        file_path = os.path.join("./figure", f'same{same_dist}_cat_{column}.png')
        fig, ax = plt.subplots()
        plt.hist([real_data, generated_data], color=['g', 'r'], label=['Real Data', 'Generated Data'])
        plt.legend()
        plt.title(title_text)
        plt.savefig(file_path)
        plt.close()

    def plot_numerical(self, column, real_data, generated_data, same_dist):
        title_text = f"{column} same dist={same_dist}"
        file_path = os.path.join("./figure", f'same{same_dist}_num_{column}.png')
        fig, ax = plt.subplots()
        plt.hist(real_data, bins=30, alpha=0.5, density=True, color='g', label='Real Data')
        plt.hist(generated_data, bins=30, alpha=0.5, density=True, color='r', label='Generated Data')
        plt.legend()
        plt.title(title_text)
        plt.savefig(file_path)
        plt.close()


# if __name__ == "__main__":

#     @ex.automain
#     def main(_config):
#         return _config
    
#     import matplotlib.pyplot as plt
#     config = main()
#     modality = "table"
#     mlm_prob = -1
#     for seed in [2021]:
#         table_real = create_label_table(config, dataset="nontest")
#         table_gen = pd.read_csv('./baseline/CTGAN.csv')

#         assert table_real.shape[1] == table_gen.shape[1]
#         print(table_gen.shape, table_real.shape)
#         stat = StatAnalysis(table_real, table_gen, config, significance_level=0.05, plot=False)
#         results = stat.compare_aki_group()
        