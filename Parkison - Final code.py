import pandas as pd
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.ensemble import IsolationForest

# Read dataset into a DataFrame
df = pd.read_csv("CDU Coding\HIT 140 Foundation Data\Ass3\po2_data.csv")

## TASK 1

# Basic Linear Regression Model (60:40 train-test split)

# Function to train and evaluate a Linear Regression model
def evaluate_linear_regression(X_train, X_test, y_train, y_test, target_name):
    print(f"********CALCULATION FOR {target_name}********")
    
    # Train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Print model parameters
    print("Intercept: ", model.intercept_)
    print("Coefficient: ", model.coef_)
    
    # Predict using the model
    y_pred = model.predict(X_test)
    
    # Compute performance metrics
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    rmse_norm = rmse / (y_test.max() - y_test.min())
    r_2 = metrics.r2_score(y_test, y_pred)
    n = len(y_test)
    p = X_test.shape[1]
    adjusted_r2 = 1 - ((1 - r_2) * (n - 1) / (n - p - 1))
    
    # Display results
    print(f"{target_name} MLP performance :")
    print("MAE: ", mae)
    print("MSE: ", mse)
    print("RMSE: ", rmse)
    print("RMSE (Normalised): ", rmse_norm)
    print("R^2: ", r_2)
    print("Adjusted R^2 :", adjusted_r2)
    
    return model
# Separate explanatory variables (x) from the dataset
x = df.drop(columns=['motor_updrs', 'total_updrs']).values

# Evaluate Linear Regression model for motor_updrs
y_motor = df['motor_updrs'].values

# Evaluate Linear Regression model for total_updrs
y_total = df['total_updrs'].values


## BASELINE MODEL 

# Function to evaluate the baseline model
def evaluate_baseline_model(y_train, y_test, target_name):
    print(f"********BASELINE FOR {target_name}********")
    
    # Use the mean of the training data as the prediction for the test data
    y_pred_base = [np.mean(y_train)] * len(y_test)
    
    # Compute performance metrics
    mae = metrics.mean_absolute_error(y_test, y_pred_base)
    mse = metrics.mean_squared_error(y_test, y_pred_base)
    rmse = math.sqrt(mse)
    rmse_norm = rmse / (y_test.max() - y_test.min())
    r_2 = metrics.r2_score(y_test, y_pred_base)
    n = len(y_test)
    p = x.shape[1]  # Using the number of independent variables from x
    adjusted_r2 = 1 - ((1 - r_2) * (n - 1) / (n - p - 1))
    
    # Display results
    print(f"{target_name} Baseline performance:")
    print("MAE: ", mae)
    print("MSE: ", mse)
    print("RMSE: ", rmse)
    print("RMSE (Normalised): ", rmse_norm)
    print("R^2: ", r_2)
    print("Adjusted R^2 :", adjusted_r2)

# Separate explanatory variables (x) from the dataset
x = df.drop(columns=['motor_updrs', 'total_updrs']).values

# Split dataset into 60% training and 40% test sets 
X_train, X_test, y_train_motor, y_test_motor = train_test_split(x, df['motor_updrs'].values, test_size=0.4, random_state=0)
X_train, X_test, y_train_total, y_test_total = train_test_split(x, df['total_updrs'].values, test_size=0.4, random_state=0)

# Evaluate Linear Regression model and Baseline model for motor_updrs
evaluate_linear_regression(X_train, X_test, y_train_motor, y_test_motor, "MOTOR UPDRS")
evaluate_baseline_model(y_train_motor, y_test_motor, "MOTOR UPDRS")

# Evaluate Linear Regression model and Baseline model for total_updrs
evaluate_linear_regression(X_train, X_test, y_train_total, y_test_total, "TOTAL UPDRS")
# evaluate_baseline_model(y_train_total, y_test_total, "TOTAL UPDRS")



## TASK 2

# 1.	Linear Regression on Original Data 

# Function to evaluate the model for different train-test splits, focusing only on the specified performance metrics
def evaluate_for_different_splits(x, y, target_name):
    # Define the split ratios
    splits = [(0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.8, 0.2)]
    
    for train_size, test_size in splits:
        print(f"\nEvaluating for {int(train_size*100)}% training and {int(test_size*100)}% test...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)
        
        # Train the Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predict using the model
        y_pred = model.predict(X_test)
        
        # Compute performance metrics
        mae = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = math.sqrt(mse)
        rmse_norm = rmse / (y_test.max() - y_test.min())
        r_2 = metrics.r2_score(y_test, y_pred)
        n = len(y_test)
        p = X_test.shape[1]
        adjusted_r2 = 1 - ((1 - r_2) * (n - 1) / (n - p - 1))
        
        # Display results
        print(f"{target_name} performance (Train: {int(train_size*100)}%, Test: {int(test_size*100)}%):")
        print("MAE: ", mae)
        print("MSE: ", mse)
        print("RMSE: ", rmse)
        print("RMSE (Normalized): ", rmse_norm)
        print("R^2: ", r_2)
        print("Adjusted R^2:", adjusted_r2)

# Analyze the performance for motor_updrs
# evaluate_for_different_splits(x, y_motor, "MOTOR UPDRS")

# Analyze the performance for total_updrs
# evaluate_for_different_splits(x, y_total, "TOTAL UPDRS")

# 2.	Outlier Detection and Removal
#Check outliers - Using Automatic Outliers Detection Algorithm in Python

from sklearn.ensemble import IsolationForest

# Separate explanatory variables (x) from the dataset
x = df.drop(columns=['motor_updrs', 'total_updrs']).values

# Initialize the IsolationForest model
iso_forest = IsolationForest(contamination=0.05, random_state=42)

# Fit the model and predict outliers
outliers = iso_forest.fit_predict(x)

# Convert the predicted values; -1 is an outlier, 1 is an inlier
outliers = [True if o == -1 else False for o in outliers]

# Add the outliers column to the dataset
df['is_outlier'] = outliers

# Display the data with the outliers marked
outlier_data = df[df['is_outlier'] == True]
# print(outlier_data.head())
# print(f"Number of detected outliers: {len(outlier_data)}")

# Remove the outliers from the dataset
df_cleaned = df[df['is_outlier'] == False].drop(columns='is_outlier')

# Display the shape of the original and cleaned dataframes
# print (df.shape, df_cleaned.shape)

# 3.   Rebuilding Linear Regression on Cleaned Data

# Separate explanatory variables (x) from the cleaned dataset
x_cleaned = df_cleaned.drop(columns=['motor_updrs', 'total_updrs']).values

# y values for motor and total UPDRS from the cleaned dataset
y_motor_cleaned = df_cleaned['motor_updrs'].values
y_total_cleaned = df_cleaned['total_updrs'].values

# Analyze the performance for motor_updrs on cleaned data
print("Performance on Cleaned Data:")
evaluate_for_different_splits(x_cleaned, y_motor_cleaned, "MOTOR UPDRS")

# Analyze the performance for total_updrs on cleaned data
evaluate_for_different_splits(x_cleaned, y_total_cleaned, "TOTAL UPDRS")


from sklearn.model_selection import KFold, cross_val_score

# 4.   Apply K-Fold Cross-Validation on the Best Split

# Define the best train-test split ratio based on the above results
best_split_ratio = 0.3  # 70% training, 30% testing

# Split the cleaned data based on the best split
X_train_motor, X_test_motor, y_train_motor, y_test_motor = train_test_split(x_cleaned, y_motor_cleaned, test_size=best_split_ratio, random_state=0)
X_train_total, X_test_total, y_train_total, y_test_total = train_test_split(x_cleaned, y_total_cleaned, test_size=best_split_ratio, random_state=0)

# Initialize the Linear Regression model
model_motor = LinearRegression()
model_total = LinearRegression()

# Apply K-Fold Cross-Validation for k=5 and k=10
k_values = [5, 10]
cross_val_results_motor = {}
cross_val_results_total = {}

for k in k_values:
    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    motor_scores = cross_val_score(model_motor, X_train_motor, y_train_motor, cv=kf, scoring='r2')
    total_scores = cross_val_score(model_total, X_train_total, y_train_total, cv=kf, scoring='r2')
    
    cross_val_results_motor[k] = motor_scores.mean()
    cross_val_results_total[k] = total_scores.mean()

# print (cross_val_results_motor, cross_val_results_total)


# Task 3: Log-transform and collinearity analysis

# Import necessary libraries for plotting
import matplotlib.pyplot as plt

# Set up the plotting figure
fig, axs = plt.subplots(4, 2, figsize=(15, 20))

# Specify the features to be plotted
features = ['jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)']

# Iterate over specified features and create scatter plots against motor_updrs and total_updrs
for i, feature in enumerate(features):
    axs[i, 0].scatter(df[feature], df['motor_updrs'], alpha=0.5)
    axs[i, 0].set_xlabel(feature)
    axs[i, 0].set_ylabel('Motor UPDRS')
    axs[i, 0].set_title(f'Motor UPDRS vs {feature}')
    
    axs[i, 1].scatter(df[feature], df['total_updrs'], alpha=0.5, color='orange')
    axs[i, 1].set_xlabel(feature)
    axs[i, 1].set_ylabel('Total UPDRS')
    axs[i, 1].set_title(f'Total UPDRS vs {feature}')

# Adjust layout
plt.tight_layout()
# plt.show()

# List of features to check
features_to_check = [
    'jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)',
    'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)',
    'shimmer(apq11)', 'nhr', 'hnr', 'rpde', 'dfa', 'ppe'
]
# Calculating skewness for each feature using the original dataframe
skewness_values_original = df[features_to_check].skew()

# Displaying skewness values from the original dataframe
# print (skewness_values_original)

# Apply log transformation to selected features
features_to_transform = ['jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)',
                         'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)',
                         'nhr', 'hnr', 'rpde', 'dfa', 'ppe']

# Create new log-transformed columns
for feature in features_to_transform:
    df[f'log_{feature}'] = df[feature].apply(lambda x: np.log(x) if x > 0 else 0)

# Visualize the effect of the transformation for the first feature 'jitter(%)' as an example
plt.figure(figsize=(20, 10))

# Original feature
plt.subplot(1, 2, 1)
plt.scatter(df['jitter(%)'], df['motor_updrs'], color="green", alpha=0.5)
plt.title("Original jitter(%)")
plt.xlabel("jitter(%)")
plt.ylabel("Motor UPDRS")

# Log-transformed feature
plt.subplot(1, 2, 2)
plt.scatter(df['log_jitter(%)'], df['motor_updrs'], color="red", alpha=0.5)
plt.title("Log Transformed jitter(%)")
plt.xlabel("log_jitter(%)")
plt.ylabel("Motor UPDRS")

plt.tight_layout()
# plt.show()

# Drop the original features that were log-transformed
df_po2_transformed = df.drop(columns=features_to_transform)

# Selecting reduced set of features for heatmap
selected_features = ['log_jitter(%)', 'log_shimmer(%)', 'log_rpde', 'log_dfa', 'log_ppe']

# Extracting the correlation matrix for the selected features
corr_selected = df_po2_transformed[selected_features].corr()

# Plotting the heatmap
import seaborn as sns
# Plot correlation matrix for the transformed dataset
corr_transformed = df_po2_transformed.corr()

# Plot the pairwise correlation as heatmap
plt.figure(figsize=(18, 16))
ax = sns.heatmap(
    corr_transformed, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True,
    annot=False
)

# Customize the labels
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)

# plt.show()



# Reduced set of features based on analysis
reduced_features = ['log_jitter(%)', 'log_rpde', 'log_dfa', 'log_ppe', 'age', 'sex', 'test_time']

# Splitting the data with reduced features
X_reduced = df_po2_transformed[reduced_features].values

# Splitting data into 70% training and 30% testing
X_train_reduced, X_test_reduced, y_train_motor_reduced, y_test_motor_reduced = train_test_split(X_reduced, df_po2_transformed['motor_updrs'].values, test_size=0.3, random_state=0)
X_train_reduced, X_test_reduced, y_train_total_reduced, y_test_total_reduced = train_test_split(X_reduced, df_po2_transformed['total_updrs'].values, test_size=0.3, random_state=0)

# Evaluate Linear Regression model for 'motor_updrs' with reduced features
print("\nPerformance on Reduced Features for MOTOR UPDRS:")
# model_motor_reduced = evaluate_linear_regression(X_train_reduced, X_test_reduced, y_train_motor_reduced, y_test_motor_reduced, "MOTOR UPDRS")

# Evaluate Linear Regression model for 'total_updrs' with reduced features
print("\nPerformance on Reduced Features for TOTAL UPDRS:")
# model_total_reduced = evaluate_linear_regression(X_train_reduced, X_test_reduced, y_train_total_reduced, y_test_total_reduced, "TOTAL UPDRS")


## TASK 4

##### STEP 1
# Step 1: Standardization of the features

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



##### STEP 2
# Step 2: The Gaussian transformation
from scipy.stats import boxcox, yeojohnson
from sklearn.preprocessing import PowerTransformer

# Check if any column has negative values
negative_values = (df_standardized[features_to_standardize.columns] < 0).sum()

# Determine if we can use Box-Cox or need to use Yeo-Johnson
transformer_type = 'box-cox' if all(negative_values == 0) else 'yeo-johnson'

# Apply the transformation Yeo-Johnson
transformer = PowerTransformer(method=transformer_type, standardize=False)
gaussian_transformed_features = transformer.fit_transform(df_standardized[features_to_standardize.columns])

# Replace original columns with transformed values
df_transformed = df_standardized.copy()
df_transformed[features_to_standardize.columns] = gaussian_transformed_features

# Display the first few rows of the transformed data
print(df_transformed.head())


##### STEP 3
# Step 3: Comparison of before and after Gaussian transformation

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
    plt.show()



##### STEP 4
# Step 4: Rebuilding the Regression Model

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

# Calculate adjusted R^2
def adjusted_r2(r2, n, k):
    return 1 - ((1 - r2) * (n - 1) / (n - k - 1))

# Building and evaluating the model for 'motor_updrs' (augmented with new metrics)
motor_model_transformed = LinearRegression().fit(X_train_transformed, y_motor_train_transformed)
motor_preds_transformed = motor_model_transformed.predict(X_test_transformed)

motor_mae_transformed = mean_absolute_error(y_motor_test_transformed, motor_preds_transformed)
motor_mse_transformed = mean_squared_error(y_motor_test_transformed, motor_preds_transformed)
motor_rmse_transformed = mean_squared_error(y_motor_test_transformed, motor_preds_transformed, squared=False)
motor_rmse_norm_transformed = motor_rmse_transformed / (y_motor_test_transformed.max() - y_motor_test_transformed.min())
motor_r2_transformed = r2_score(y_motor_test_transformed, motor_preds_transformed)
motor_adj_r2_transformed = adjusted_r2(motor_r2_transformed, len(y_motor_test_transformed), X_transformed.shape[1])

# Building and evaluating the model for 'total_updrs' (augmented with new metrics)
total_model_transformed = LinearRegression().fit(X_train_transformed, y_total_train_transformed)
total_preds_transformed = total_model_transformed.predict(X_test_transformed)

total_mae_transformed = mean_absolute_error(y_total_test_transformed, total_preds_transformed)
total_mse_transformed = mean_squared_error(y_total_test_transformed, total_preds_transformed)
total_rmse_transformed = mean_squared_error(y_total_test_transformed, total_preds_transformed, squared=False)
total_rmse_norm_transformed = total_rmse_transformed / (y_total_test_transformed.max() - y_total_test_transformed.min())
total_r2_transformed = r2_score(y_total_test_transformed, total_preds_transformed)
total_adj_r2_transformed = adjusted_r2(total_r2_transformed, len(y_total_test_transformed), X_transformed.shape[1])



# Displaying the metrics for motor_updrs
print("Linear Regression performance for motor_updrs:")
print("MAE: ", motor_mae_transformed)
print("MSE: ", motor_mse_transformed)
print("RMSE: ", motor_rmse_transformed)
print("RMSE (Normalised): ", motor_rmse_norm_transformed)
print("R^2: ", motor_r2_transformed)
print("Adjusted R^2: ", motor_adj_r2_transformed)
print("\n")

# Displaying the metrics for total_updrs
print("Linear Regression performance for total_updrs:")
print("MAE: ", total_mae_transformed)
print("MSE: ", total_mse_transformed)
print("RMSE: ", total_rmse_transformed)
print("RMSE (Normalised): ", total_rmse_norm_transformed)
print("R^2: ", total_r2_transformed)
print("Adjusted R^2: ", total_adj_r2_transformed)
