import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('../data/user_behavior_dataset.csv')

X = df.drop(columns=["User ID", "User Behavior Class"], errors='ignore')

categorical_columns = ["Device Model", "Operating System", "Gender"]
numeric_columns = [col for col in X.columns if col not in categorical_columns]

ohe = OneHotEncoder(drop='first', sparse_output=False)  
X_cat = ohe.fit_transform(X[categorical_columns])

X_num = X[numeric_columns].values
X_combined = np.hstack([X_num, X_cat])

categorical_column_names = ohe.get_feature_names_out(categorical_columns)
combined_columns = numeric_columns + list(categorical_column_names)

X_combined_df = pd.DataFrame(X_combined, columns=combined_columns)

output_path = '../data/transformed_user_behavior_dataset.csv'
X_combined_df.to_csv(output_path, index=False)
