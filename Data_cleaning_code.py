import pandas as pd

# Load the dataset
df_cleaned = pd.read_csv("CDU Coding\HIT 140 Foundation Data\Ass3\po2_data_cleaned.csv")


#### STEP 1
import matplotlib.pyplot as plt
import seaborn as sns

# Calculate the correlation matrix
correlation_matrix = df_cleaned.corr()

# Refining the Correlation Analysis
excluded_features = ['age', 'subject#']
correlation_matrix_refined = correlation_matrix.drop(index=excluded_features, columns=excluded_features)

top_motor_updrs_refined = correlation_matrix_refined['motor_updrs'].drop(['motor_updrs', 'total_updrs']).sort_values(ascending=False).head(5)
top_total_updrs_refined = correlation_matrix_refined['total_updrs'].drop(['motor_updrs', 'total_updrs']).sort_values(ascending=False).head(5)

# Visualizing the top correlated features for 'motor_updrs' after refining
plt.figure(figsize=(10, 6))
sns.barplot(x=top_motor_updrs_refined.index, y=top_motor_updrs_refined.values, palette="viridis")
plt.title('Top 5 Features Correlated with motor_updrs')
plt.ylabel('Correlation Coefficient')
plt.xticks(rotation=45)
plt.tight_layout()
# plt.show()

# Visualizing the top correlated features for 'total_updrs' after refining
plt.figure(figsize=(10, 6))
sns.barplot(x=top_total_updrs_refined.index, y=top_total_updrs_refined.values, palette="viridis")
plt.title('Top 5 Features Correlated with total_updrs')
plt.ylabel('Correlation Coefficient')
plt.xticks(rotation=45)
plt.tight_layout()
# plt.show()

# print(top_motor_updrs_refined)
# print(top_total_updrs_refined)



##### STEP 2
# Step 2: Standardization of the features

from sklearn.preprocessing import StandardScaler

# Excluding non-voice metric features for standardization
features_to_standardize = df_cleaned.drop(columns=['age', 'subject#', 'motor_updrs', 'total_updrs'])

# Standardizing the features
scaler = StandardScaler()
standardized_features = scaler.fit_transform(features_to_standardize)

# Replace original columns with standardized values
df_standardized = df_cleaned.copy()
df_standardized[features_to_standardize.columns] = standardized_features

# Display the first few rows of the standardized data
print(df_standardized.head())



##### The Gaussian transformation
from scipy.stats import boxcox, yeojohnson
from sklearn.preprocessing import PowerTransformer

# Check if any column has negative values
negative_values = (df_standardized[features_to_standardize.columns] < 0).sum()

# Determine if we can use Box-Cox or need to use Yeo-Johnson
transformer_type = 'box-cox' if all(negative_values == 0) else 'yeo-johnson'

# Apply the transformation
transformer = PowerTransformer(method=transformer_type, standardize=False)
gaussian_transformed_features = transformer.fit_transform(df_standardized[features_to_standardize.columns])

# Replace original columns with transformed values
df_transformed = df_standardized.copy()
df_transformed[features_to_standardize.columns] = gaussian_transformed_features

# Display the first few rows of the transformed data
print(df_transformed.head())


### Comparison of before and after Gaussian transformation

import numpy as np
excluded_features = ['sex', 'test_time']
refined_features = [feature for feature in features_to_standardize.columns if feature not in excluded_features]

# Dividing the refined features into groups of 3 for separate plots
feature_groups_refined = [refined_features[i:i+3] for i in range(0, len(refined_features), 3)]

for group in feature_groups_refined:
    fig, axes = plt.subplots(len(group), 2, figsize=(15, 4 * len(group)))
    
    # If there's only one feature, adjust the axes to be 1D array for consistency
    if len(group) == 1:
        axes = np.expand_dims(axes, axis=0)
    
    for i, feature in enumerate(group):
        # Before transformation
        sns.kdeplot(df_standardized[feature], ax=axes[i, 0], shade=True, color="b")
        axes[i, 0].set_title(f"{feature} - Before Transformation")
        axes[i, 0].set_ylabel("Density")
        
        # After transformation
        sns.kdeplot(df_transformed[feature], ax=axes[i, 1], shade=True, color="r")
        axes[i, 1].set_title(f"{feature} - After Transformation")
        axes[i, 1].set_ylabel("Density")
    
    plt.tight_layout()
    # plt.show()





##### STEP 4
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Combining the original code with the new metric calculations

# Split the transformed data into 70% training and 30% test sets
X_transformed = df_transformed.drop(columns=['motor_updrs', 'total_updrs'])
y_motor_transformed = df_transformed['motor_updrs']
y_total_transformed = df_transformed['total_updrs']

X_train_transformed, X_test_transformed, y_motor_train_transformed, y_motor_test_transformed = train_test_split(X_transformed, y_motor_transformed, test_size=0.3, random_state=42)
_, _, y_total_train_transformed, y_total_test_transformed = train_test_split(X_transformed, y_total_transformed, test_size=0.3, random_state=42)


# Building and evaluating the model for 'motor_updrs' (augmented with new metrics)
motor_model_transformed = LinearRegression().fit(X_train_transformed, y_motor_train_transformed)
motor_preds_transformed = motor_model_transformed.predict(X_test_transformed)

motor_mae_transformed = mean_absolute_error(y_motor_test_transformed, motor_preds_transformed)
motor_mse_transformed = mean_squared_error(y_motor_test_transformed, motor_preds_transformed)
motor_rmse_transformed = mean_squared_error(y_motor_test_transformed, motor_preds_transformed, squared=False)
motor_rmse_norm_transformed = motor_rmse_transformed / (y_motor_test_transformed.max() - y_motor_test_transformed.min())
motor_r2_transformed = r2_score(y_motor_test_transformed, motor_preds_transformed)

# Building and evaluating the model for 'total_updrs' (augmented with new metrics)
total_model_transformed = LinearRegression().fit(X_train_transformed, y_total_train_transformed)
total_preds_transformed = total_model_transformed.predict(X_test_transformed)

total_mae_transformed = mean_absolute_error(y_total_test_transformed, total_preds_transformed)
total_mse_transformed = mean_squared_error(y_total_test_transformed, total_preds_transformed)
total_rmse_transformed = mean_squared_error(y_total_test_transformed, total_preds_transformed, squared=False)
total_rmse_norm_transformed = total_rmse_transformed / (y_total_test_transformed.max() - y_total_test_transformed.min())
total_r2_transformed = r2_score(y_total_test_transformed, total_preds_transformed)


# Displaying the metrics for motor_updrs
print("Linear Regression performance for motor_updrs:")
print("MAE: ", motor_mae_transformed)
print("MSE: ", motor_mse_transformed)
print("RMSE: ", motor_rmse_transformed)
print("RMSE (Normalised): ", motor_rmse_norm_transformed)
print("R^2: ", motor_r2_transformed)
print("\n")

# Displaying the metrics for total_updrs
print("Linear Regression performance for total_updrs:")
print("MAE: ", total_mae_transformed)
print("MSE: ", total_mse_transformed)
print("RMSE: ", total_rmse_transformed)
print("RMSE (Normalised): ", total_rmse_norm_transformed)
print("R^2: ", total_r2_transformed)
