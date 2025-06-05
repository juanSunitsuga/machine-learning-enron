# import joblib
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, accuracy_score
# from sklearn.impute import SimpleImputer
# from collections import Counter

# # Load cleaned dataset
# data_dict = joblib.load(open("D:\\College\\Semester IV\\ML\\Tubes\\machine-learning-enron\\final_project_dataset_cleaned.pkl", "rb"))

# # Define features
# finance_feature_list = [
#     'salary', 'bonus', 'total_payments', 'long_term_incentive',
#     'total_stock_value', 'exercised_stock_options', 'expenses'
# ]
# email_feature_list = [
#     'from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi',
#     'fraction_from_poi', 'fraction_to_poi', 'fraction_shared_receipt_with_poi'
# ]
# all_features = finance_feature_list + email_feature_list

# # Prepare raw data arrays with np.nan for missing
# X = []
# y = []
# for name, features in data_dict.items():
#     row = []
#     for f in all_features:
#         val = features.get(f, 'NaN')
#         row.append(float(val) if val != 'NaN' else np.nan)
#     X.append(row)
#     y.append(int(features.get('poi', 0)))
# X = np.array(X)
# y = np.array(y)

# print(f"Dataset size: {X.shape[0]} samples, {X.shape[1]} features")
# print("Class distribution:", Counter(y))

# # Impute missing values (mean)
# imputer = SimpleImputer(strategy='mean')
# X_imputed = imputer.fit_transform(X)

# # Split data with stratification to keep POI ratio consistent
# X_train, X_test, y_train, y_test = train_test_split(
#     X_imputed, y, test_size=0.3, random_state=42, stratify=y)

# # Train Random Forest classifier
# clf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf.fit(X_train, y_train)

# # Predict and evaluate
# y_pred = clf.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Classification Report:\n", classification_report(y_test, y_pred))

# # Save the trained model
# joblib.dump(clf, "random_forest_enron.pkl")

import joblib
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load your cleaned dataset
data_dict = joblib.load(open("D:\\College\\Semester IV\\ML\\Tubes\\machine-learning-enron\\final_project_dataset_cleaned.pkl", "rb"))

# Define features (include 'poi' as the first one!)
finance_feature_list = [
    'salary', 'bonus', 'total_payments', 'long_term_incentive',
    'total_stock_value', 'expenses'
]
email_feature_list = [
    'from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi'
]

features_list = ['poi'] + finance_feature_list + email_feature_list  # required order

# Prepare data
X = []
y = []
for name, features in data_dict.items():
    row = []
    for f in features_list[1:]:  # skip 'poi'
        val = features.get(f, 'NaN')
        row.append(float(val) if val != 'NaN' else 0.0)
    X.append(row)
    y.append(int(features.get('poi', 0)))

X = np.array(X)
y = np.array(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save required .pkl files
with open("my_classifier.pkl", "wb") as clf_out:
    pickle.dump(clf, clf_out)

with open("my_dataset.pkl", "wb") as dataset_out:
    pickle.dump(data_dict, dataset_out)

with open("my_feature_list.pkl", "wb") as features_out:
    pickle.dump(features_list, features_out)
