import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import csv
from sklearn.linear_model import LinearRegression

# Load dataset
data_dict = joblib.load(open("D:\Perkuliahan\File Kuliah Semester 4\ML\Tubes\machine-learning-enron\\final_project_dataset.pkl", "rb"))

print(f"Original dataset size: {len(data_dict)}")

# Define features
finance_feature_list = [
    'salary', 'bonus', 'total_payments', 'long_term_incentive',
    'total_stock_value', 'exercised_stock_options', 'expenses'
]
email_feature_list = [
    'from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi',
    'fraction_from_poi', 'fraction_to_poi', 'fraction_shared_receipt_with_poi'
]

# Remove known non-person outliers
outliers_to_remove = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']
for outlier in outliers_to_remove:
    if outlier in data_dict:
        print(f"Removing: {outlier}")
        data_dict.pop(outlier)

print(f"After removing non-person entries: {len(data_dict)}")

# Create fraction features for email data
print("Creating email fraction features...")
for name, features in data_dict.items():
    # Fraction of emails received from POI
    if features['to_messages'] != 'NaN' and features['from_poi_to_this_person'] != 'NaN' and float(features['to_messages']) > 0:
        features['fraction_from_poi'] = float(features['from_poi_to_this_person']) / float(features['to_messages'])
    else:
        features['fraction_from_poi'] = 0.0

    # Fraction of emails sent to POI
    if features['from_messages'] != 'NaN' and features['from_this_person_to_poi'] != 'NaN' and float(features['from_messages']) > 0:
        features['fraction_to_poi'] = float(features['from_this_person_to_poi']) / float(features['from_messages'])
    else:
        features['fraction_to_poi'] = 0.0

    # Fraction of shared receipts with POI
    if features['to_messages'] != 'NaN' and features['shared_receipt_with_poi'] != 'NaN' and float(features['to_messages']) > 0:
        features['fraction_shared_receipt_with_poi'] = float(features['shared_receipt_with_poi']) / float(features['to_messages'])
    else:
        features['fraction_shared_receipt_with_poi'] = 0.0

all_features = finance_feature_list + email_feature_list

# Remove entries with too many NaNs
min_non_nan = 3
names_to_remove = []
for name, features in data_dict.items():
    non_nan_count = sum(1 for f in all_features if features.get(f, 'NaN') != 'NaN')
    if non_nan_count < min_non_nan:
        names_to_remove.append(name)

for name in names_to_remove:
    print(f"Removing (too many NaNs): {name}")
    data_dict.pop(name, None)

print(f"After removing entries with too many NaNs: {len(data_dict)}")

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
print(f"Feature matrix shape: {feature_matrix.shape}")

# Scale features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(feature_matrix)

# Detect outliers using IQR for each feature (but don't remove automatically)
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

print(f"\nPotential outliers detected (after scaling): {len(outlier_names)}")
for name in sorted(outlier_names):
    print(f"  {name}")

# Count POIs in the dataset
poi_count = sum(1 for features in data_dict.values() if features.get('poi', 0) == 1)
print(f"\nPOI count in dataset: {poi_count}")
print(f"Non-POI count: {len(data_dict) - poi_count}")

# --- Clean finance features using IQR (conservative approach) ---
print("\nCleaning finance features using IQR...")
finance_outlier_names = set()
for i, feature in enumerate(finance_feature_list):
    col = scaled_features[:, i]
    q1 = np.percentile(col, 25)
    q3 = np.percentile(col, 75)
    iqr = q3 - q1
    lower = q1 - 2.0 * iqr  # More conservative threshold
    upper = q3 + 2.0 * iqr
    for idx, val in enumerate(col):
        if val < lower or val > upper:
            finance_outlier_names.add(names_list[idx])

print(f"Finance outliers to remove: {len(finance_outlier_names)}")
for name in sorted(finance_outlier_names):
    print(f"  {name}")

# Remove finance outliers (only if they are clearly data errors, not POIs)
for name in finance_outlier_names:
    # Check if this person is a POI before removing
    if data_dict.get(name, {}).get('poi', 0) == 0:  # Only remove non-POIs
        data_dict.pop(name, None)

print(f"After finance outlier removal: {len(data_dict)}")

# Save cleaned dataset
my_feature_list = ['poi'] + all_features

# Save as pickle files for the project evaluator
with open("my_dataset.pkl", "wb") as f:
    pickle.dump(data_dict, f)

with open("my_feature_list.pkl", "wb") as f:
    pickle.dump(my_feature_list, f)

# Also save as the cleaned dataset file
joblib.dump(data_dict, "D:\Perkuliahan\File Kuliah Semester 4\ML\Tubes\machine-learning-enron\\final_project_dataset_cleaned.pkl")

# Convert to CSV for easy viewing
with open("final_project_dataset_cleaned.csv", "w", newline='') as f:
    writer = csv.writer(f)
    # Write header
    header = ["name"] + my_feature_list
    writer.writerow(header)
    # Write rows
    for name, features in data_dict.items():
        row = [name] + [features.get(k, "NaN") for k in my_feature_list]
        writer.writerow(row)

print(f"\nFinal cleaned dataset size: {len(data_dict)}")
print("Files created:")
print("  - my_dataset.pkl (for project evaluator)")
print("  - my_feature_list.pkl (for project evaluator)")
print("  - final_project_dataset_cleaned.pkl (backup)")
print("  - final_project_dataset_cleaned.csv (for viewing)")

# Final POI count check
final_poi_count = sum(1 for features in data_dict.values() if features.get('poi', 0) == 1)
print(f"\nFinal POI count: {final_poi_count}")
print(f"Final Non-POI count: {len(data_dict) - final_poi_count}")

# import pickle
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.feature_selection import SelectKBest, f_classif
# import csv

# # === Load dataset ===
# import joblib
# data_dict = joblib.load("D:\\College\\Semester IV\\ML\\Tubes\\machine-learning-enron\\final_project_dataset.pkl")

# print(f"Original dataset size: {len(data_dict)}")

# # === Remove outliers ===
# for outlier in ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']:
#     if outlier in data_dict:
#         print(f"Removing: {outlier}")
#         data_dict.pop(outlier)

# print(f"After removing known outliers: {len(data_dict)}")

# # === Define features ===
# finance_features = [
#     'salary', 'bonus', 'total_payments', 'long_term_incentive',
#     'total_stock_value', 'exercised_stock_options', 'expenses'
# ]
# email_features = [
#     'from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi',
#     'fraction_from_poi', 'fraction_to_poi', 'fraction_shared_receipt_with_poi'
# ]
# all_features = finance_features + email_features

# # === Create fraction features ===
# for person in data_dict.values():
#     to_msgs = person.get('to_messages', 'NaN')
#     from_poi = person.get('from_poi_to_this_person', 'NaN')
#     from_msgs = person.get('from_messages', 'NaN')
#     to_poi = person.get('from_this_person_to_poi', 'NaN')
#     shared = person.get('shared_receipt_with_poi', 'NaN')

#     person['fraction_from_poi'] = float(from_poi) / float(to_msgs) if to_msgs != 'NaN' and from_poi != 'NaN' else 0.0
#     person['fraction_to_poi'] = float(to_poi) / float(from_msgs) if from_msgs != 'NaN' and to_poi != 'NaN' else 0.0
#     person['fraction_shared_receipt_with_poi'] = float(shared) / float(to_msgs) if to_msgs != 'NaN' and shared != 'NaN' else 0.0

# # === Remove people with too many NaNs ===
# min_non_nan = 3
# to_remove = []
# for name, features in data_dict.items():
#     non_nan_count = sum(1 for f in all_features if features.get(f, 'NaN') != 'NaN')
#     if non_nan_count < min_non_nan:
#         to_remove.append(name)

# for name in to_remove:
#     print(f"Removing {name} (too many NaNs)")
#     data_dict.pop(name)

# print(f"After cleaning for NaNs: {len(data_dict)}")

# # === Prepare features and labels ===
# feature_matrix = []
# labels = []
# names = []
# for name, person in data_dict.items():
#     row = [float(person.get(f, 0.0)) if person.get(f, 'NaN') != 'NaN' else 0.0 for f in all_features]
#     feature_matrix.append(row)
#     labels.append(int(person.get('poi', 0)))
#     names.append(name)

# feature_matrix = np.array(feature_matrix)
# labels = np.array(labels)

# # === Scale features ===
# scaler = MinMaxScaler()
# scaled_features = scaler.fit_transform(feature_matrix)

# # === SelectKBest ===
# selector = SelectKBest(score_func=f_classif, k='all')
# selector.fit(scaled_features, labels)

# # === Show feature scores ===
# print("\nFeature scores from SelectKBest:")
# feature_scores = list(zip(all_features, selector.scores_))
# sorted_scores = sorted(feature_scores, key=lambda x: x[1], reverse=True)
# for f, score in sorted_scores:
#     print(f"{f}: {score:.4f}")

# # === Remove finance outliers conservatively (IQR) ===
# finance_outlier_names = set()
# for i, feature in enumerate(finance_features):
#     col = scaled_features[:, i]
#     q1 = np.percentile(col, 25)
#     q3 = np.percentile(col, 75)
#     iqr = q3 - q1
#     lower = q1 - 2 * iqr
#     upper = q3 + 2 * iqr
#     for idx, val in enumerate(col):
#         if val < lower or val > upper:
#             if labels[idx] == 0:
#                 finance_outlier_names.add(names[idx])

# for name in finance_outlier_names:
#     print(f"Removing finance outlier: {name}")
#     data_dict.pop(name, None)

# print(f"After finance outlier removal: {len(data_dict)}")

# # === Save final dataset ===
# my_feature_list = ['poi'] + all_features

# with open("my_dataset.pkl", "wb") as f:
#     pickle.dump(data_dict, f)

# with open("my_feature_list.pkl", "wb") as f:
#     pickle.dump(my_feature_list, f)

# with open("final_project_dataset_cleaned.csv", "w", newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(['name'] + my_feature_list)
#     for name, features in data_dict.items():
#         row = [name] + [features.get(k, 'NaN') for k in my_feature_list]
#         writer.writerow(row)

# print("\nCleaned dataset saved:")
# print("  - my_dataset.pkl")
# print("  - my_feature_list.pkl")
# print("  - final_project_dataset_cleaned.csv")

# # === Final POI count ===
# final_poi_count = sum(1 for person in data_dict.values() if person.get('poi', 0) == 1)
# print(f"\nFinal POI count: {final_poi_count}")
# print(f"Final Non-POI count: {len(data_dict) - final_poi_count}")
