"""
code for get benchmark dataset and features calculation by Magpie
"""
import os
import pandas as pd
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf
from matminer.datasets import load_dataset

if __name__ == '__main__':
    datasets_info = [
        {"dataset_name": 'steel_strength', 'target_col': "tensile strength"},
        {"dataset_name": 'brgoch_superhard_training', 'target_col': "shear_modulus"},
        {"dataset_name": 'double_perovskites_gap', 'target_col': 'gap gllbsc'},
        {"dataset_name": 'superconductivity2018', 'target_col': 'Tc'},
        {"dataset_name": 'matbench_expt_gap', 'target_col': 'gap expt'},
        {"dataset_name": 'castelli_perovskites', 'target_col': 'e_form'},
        {"dataset_name": 'expt_formation_enthalpy', 'target_col': "e_form expt"},
        {"dataset_name": 'expt_formation_enthalpy_kingsbury', 'target_col': "expt_form_e"},
        {"dataset_name": 'expt_gap', 'target_col': "gap expt"},
        {"dataset_name": 'wolverton_oxides', 'target_col': "e_form"}
    ]
    result_info = {
    }
    result_df = pd.DataFrame()
    Y_col = 'target'

    for info in datasets_info:
        dataset_name = info['dataset_name']
        target_col = info['target_col']

        data_file_name = f"./data/{dataset_name}_{target_col}.csv"
        if os.path.exists(data_file_name):
            ml_dataset = pd.read_csv(data_file_name)
            features = list(ml_dataset.columns[:-1])
            print(f" {dataset_name} data size:", len(ml_dataset))
        else:
            # save dataset with calculated features
            df = load_dataset(dataset_name)
            # add magpie features
            if 'formula' in list(df.columns):
                formula_col = 'formula'
            else:
                formula_col = 'composition'
            df_chemistry_formula = pd.DataFrame({"formula": df[formula_col], "target": df[target_col]})
            df_magpie = StrToComposition(target_col_id='composition_obj').featurize_dataframe(df_chemistry_formula,
                                                                                              'formula',
                                                                                              ignore_errors=True)
            feature_calculators = MultipleFeaturizer([cf.Stoichiometry(), cf.ElementProperty.from_preset("magpie"),
                                                      cf.ValenceOrbital(props=['avg']), cf.IonProperty(fast=True)])
            feature_labels = feature_calculators.feature_labels()
            df_magpie = feature_calculators.featurize_dataframe(df_magpie, col_id='composition_obj', ignore_errors=True)
            features = list(df_magpie.columns[3:])
            ml_dataset = df_magpie[features + ['target']].dropna()
            ml_dataset.to_csv(data_file_name, index=False)
            print(f"{data_file_name} save to disk")
