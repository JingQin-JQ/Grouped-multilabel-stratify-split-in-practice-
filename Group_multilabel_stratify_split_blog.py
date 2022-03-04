# Databricks notebook source
# imports
import os
import pandas as pd
import numpy as np
import shap
from pyspark.sql import functions as F
from pyspark.sql import types as t
import seaborn as sns

import sklearn

from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import skmultilearn.model_selection as ms


import matplotlib

import pickle
import xgboost as xgb


import itertools
import datetime
from pyspark.sql.window import Window

# COMMAND ----------

def get_linked_id( df, cust_rel, N_iter):
        df = df[['id']].drop_duplicates()
        df['linked_id'] = df['id'].copy()

        print(f'Initial number of unique Ids: {df.shape[0]}')
        for i in range(N_iter):
            df = expand_single_step(df, cust_rel)
            df_linked = df.sort_values(['id', 'linked_id']).drop_duplicates(subset='id', keep='first')
            n_unique_groups = df_linked['linked_id'].unique().shape[0]
            print(f'Number of unique Id groups after iteration {i + 1}: {n_unique_groups}')

        ind = df_linked["linked_id"].isna()
        df_linked.loc[ind, 'linked_id'] = df_linked.loc[ind, 'id']
        return df_linked

def expand_single_step(df, cust_rel):
        df = df.merge(cust_rel, left_on='linked_id', right_on='owner_cust_id', how='left')[
            ['id', 'linked_id', 'account_cust_id']]
       # link owner_id back to account_id
        df = df.merge(cust_rel, on='account_cust_id', how='left')[['id', 'owner_cust_id']]
        df = df.rename(columns={'owner_cust_id': 'linked_id'})
        return df.drop_duplicates()


# COMMAND ----------

cust_rel = [
            ["1", "1", "IND"],["2", "2", "IND"],["3", "3", "IND"], ["4", "4", "IND"], ["5", "5", "IND"],
            ["6", "6", "IND"], ["7", "7", "IND"], ["8", "8", "IND"],  ["9", "9", "IND"], ["10", "10", "IND"],
            ["2", "3", "JOINT"], ["2", "4", "JOINT"],["3", "4", "JOINT"], ["4", "5", "JOINT"],
            ["6", "7", "JOINT"],
            ["8", "9", "JOINT"], ["8", "10", "JOINT"], ["9", "10", "JOINT"],
        ]
cust_rel_columns = ["owner_cust_id", "account_cust_id", "relationship"]

df = [
            ["1", "F", "1"], ["2", "F", "0"],["3", "M", "1"], ["4", "M", "0"], ["5", "M", "1"],
            ["6", "M", "0"], ["7", "M", "0"], ["8", "F", "1"], ["9", "F", "1"], ["10", "M", "1"],
        ]
df_struct_type = [ "id","Gender","Adult"]

# COMMAND ----------

cust_relDF =pd.DataFrame(cust_rel, columns=cust_rel_columns)

df =pd.DataFrame(df, columns=df_struct_type)


# COMMAND ----------

df_linkedid = get_linked_id(df, cust_relDF,3)

# COMMAND ----------

df_linkedid_nodu = df_linkedid.drop_duplicates('linked_id', keep="last")
df_linkedid_du = df_linkedid.drop(df_linkedid_nodu.index)

# COMMAND ----------

display(df_linkedid_nodu)

# COMMAND ----------

display(df_linkedid_du)

# COMMAND ----------

df_group = df_linkedid_nodu.merge(df, on='id', how='left')
display(df_group)

# COMMAND ----------

def split_train_test(ft_df: pd.DataFrame, test_size, seed, stratify_on):
  
  if seed is not None: np.random.seed(seed)
  df = ft_df.copy()
  stratify_lst = stratify_on.copy()

  col_object = df[stratify_lst].select_dtypes(include=[object]).columns
  tmp_cols = []
  for col in col_object:
    df[f"{col}Tmp"] = df[col].astype('category').cat.codes
    stratify_lst.remove(col)
    stratify_lst.append(f"{col}Tmp")
    tmp_cols.append(f"{col}Tmp")

  # Stratefied split
  x_train, _, x_test, _ = ms.iterative_train_test_split(df.values, df[stratify_lst].values, test_size = test_size)

  df_train = pd.DataFrame(x_train, columns = df.columns).drop(tmp_cols, axis = 1)
  df_test = pd.DataFrame(x_test, columns = df.columns).drop(tmp_cols, axis = 1)

  return df_train, df_test

df_group = df_linkedid_nodu.merge(df, on='id', how='left')
train, test = split_train_test(df_group, test_size=0.51, seed=1, stratify_on=["Gender","Adult"])

# COMMAND ----------

train, test = split_train_test(df_group, test_size=0.51, seed=1, stratify_on=["Gender","Adult"])

# COMMAND ----------

display(train)

# COMMAND ----------

display(test)

# COMMAND ----------

test_du = df_linkedid_du[df_linkedid_du['linked_id'].isin(test['linked_id'])].merge(df, on='id', how='left')
test_plus = pd.concat([test, test_du])
train_du = df_linkedid_du[df_linkedid_du['linked_id'].isin(train['linked_id'])].merge(df, on='id', how='left')
train_plus = pd.concat([train, train_du])

# COMMAND ----------

display(train_plus)

# COMMAND ----------

display(test_plus)

# COMMAND ----------

df_train = train_plus.assign(Split = "train")
df_test = test_plus.assign(Split = "test")
combine = pd.concat([df_train, df_test])
pd.crosstab(combine['Split'], combine['Gender'])

# COMMAND ----------

pd.crosstab(combine['Split'], combine['Adult'])

# COMMAND ----------


