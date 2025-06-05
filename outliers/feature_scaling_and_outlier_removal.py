import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import csv
from sklearn.linear_model import LinearRegression

# Load dataset
data_dict = joblib.load(open("c:\\Users\\juank\\Documents\\KULIAH\\Semester 4\\Machine Learning\\Tugas Besar\\final_project\\final_project_dataset.pkl", "rb"))

# Define features
finance_feature_list = [
    'salary', 'bonus', 'total_payments', 'long_term_incentive',
    'total_stock_value', 'exercised_stock_options', 'expenses'
]
email_feature_list = [
    'from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi',
    'fraction_from_poi', 'fraction_to_poi', 'fraction_shared_receipt_with_poi'
]
all_features = finance_feature_list + email_feature_list

# Remove known non-person outliers
outliers_to_remove = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']
for outlier in outliers_to_remove:
    data_dict.pop(outlier, None)

# Remove entries with too many NaNs
min_non_nan = 3
names_to_remove = []
for name, features in data_dict.items():
    non_nan_count = sum(1 for f in all_features if features.get(f, 'NaN') != 'NaN')
    if non_nan_count < min_non_nan:
        names_to_remove.append(name)
for name in names_to_remove:
    data_dict.pop(name, None)

# Prepare data for scaling and outlier detection
feature_matrix = []
names_list = []
for name, features in data_dict.items():
    row = []
    for f in all_features:
        val = features.get(f, 'NaN')
        row.append(float(val) if val != 'NaN' else 0.0)
    feature_matrix.append(row)
    names_list.append(name)
feature_matrix = np.array(feature_matrix)

# Scale features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(feature_matrix)

# Detect outliers using IQR for each feature
outlier_names = set()
for i, feature in enumerate(all_features):
    col = scaled_features[:, i]
    q1 = np.percentile(col, 25)
    q3 = np.percentile(col, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    for idx, val in enumerate(col):
        if val < lower or val > upper:
            outlier_names.add(names_list[idx])

print("Potential outliers detected (after scaling):")
for name in outlier_names:
    print(name)

# Optionally, remove detected outliers
for name in outlier_names:
    data_dict.pop(name, None)

# --- Clean finance features using IQR ---
finance_outlier_names = set()
for i, feature in enumerate(finance_feature_list):
    col = scaled_features[:, i]
    q1 = np.percentile(col, 25)
    q3 = np.percentile(col, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    for idx, val in enumerate(col):
        if val < lower or val > upper:
            finance_outlier_names.add(names_list[idx])

# Remove finance outliers
for name in finance_outlier_names:
    data_dict.pop(name, None)

# --- Clean email features using regression ---
email_outlier_names = set()
for i, feature in enumerate(email_feature_list):
    col = scaled_features[:, len(finance_feature_list) + i]
    # Use all other email features as predictors
    X = np.delete(scaled_features[:, len(finance_feature_list):], i, axis=1)
    y = col
    reg = LinearRegression()
    reg.fit(X, y)
    y_pred = reg.predict(X)
    residuals = y - y_pred
    std_res = np.std(residuals)
    for idx, res in enumerate(residuals):
        if abs(res) > 3 * std_res:
            email_outlier_names.add(names_list[idx])

# Remove email outliers
for name in email_outlier_names:
    data_dict.pop(name, None)

# Save cleaned and scaled data (optional)
joblib.dump(data_dict, "c:\\Users\\juank\\Documents\\KULIAH\\Semester 4\\Machine Learning\\Tugas Besar\\final_project\\final_project_dataset_cleaned.pkl")

with open("final_project_dataset_modified.pkl", "rb") as f:
    data = pickle.load(f)

with open("final_project_dataset_modified.csv", "w", newline='') as f:
    writer = csv.writer(f)
    # Write header
    header = ["name"] + list(next(iter(data.values())).keys())
    writer.writerow(header)
    # Write rows
    for name, features in data.items():
        row = [name] + [features.get(k, "NaN") for k in header[1:]]
        writer.writerow(row)