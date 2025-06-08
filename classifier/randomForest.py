# import joblib
# import numpy as np
# import pickle
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, accuracy_score

# # Load your cleaned dataset
# data_dict = joblib.load(open("D:\\College\\Semester IV\\ML\\Tubes\\machine-learning-enron\\final_project_dataset_cleaned.pkl", "rb"))

# # Define features (include 'poi' as the first one!)
# finance_feature_list = [
#     'salary', 'bonus', 'total_payments', 'long_term_incentive',
#     'total_stock_value', 'expenses'
# ]
# email_feature_list = [
#     'from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi'
# ]

# features_list = ['poi'] + finance_feature_list + email_feature_list  # required order

# # Prepare data
# X = []
# y = []
# for name, features in data_dict.items():
#     row = []
#     for f in features_list[1:]:  # skip 'poi'
#         val = features.get(f, 'NaN')
#         row.append(float(val) if val != 'NaN' else 0.0)
#     X.append(row)
#     y.append(int(features.get('poi', 0)))

# X = np.array(X)
# y = np.array(y)

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Train classifier
# clf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf.fit(X_train, y_train)

# # Evaluate
# y_pred = clf.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Classification Report:\n", classification_report(y_test, y_pred))

# # Save required .pkl files
# with open("my_classifier.pkl", "wb") as clf_out:
#     pickle.dump(clf, clf_out)

# with open("my_dataset.pkl", "wb") as dataset_out:
#     pickle.dump(data_dict, dataset_out)

# with open("my_feature_list.pkl", "wb") as features_out:
#     pickle.dump(features_list, features_out)

import joblib
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer

# Load cleaned dataset
data_dict = joblib.load(open("D:\\College\\Semester IV\\ML\\Tubes\\machine-learning-enron\\final_project_dataset_cleaned.pkl", "rb"))

# Define features
# Define features
finance_feature_list = [
    'salary', 'bonus', 'total_payments', 'long_term_incentive',
    'total_stock_value', 'exercised_stock_options', 'expenses'
]
email_feature_list = [
    'from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi',
    'fraction_from_poi', 'fraction_to_poi', 'fraction_shared_receipt_with_poi'
]

features_list = ['poi'] + finance_feature_list + email_feature_list

# Prepare data
X, y = [], []
for name, features in data_dict.items():
    row = []
    for f in features_list[1:]:  # skip 'poi'
        val = features.get(f, 'NaN')
        row.append(float(val) if val != 'NaN' else np.nan)
    X.append(row)
    y.append(int(features.get('poi', 0)))

X = np.array(X)
y = np.array(y)

# Impute missing values
X = SimpleImputer(strategy='mean').fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Define expanded parameter grid
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 3],
    'max_features': ['auto', 'sqrt', 'log2']
}

# GridSearchCV with class_weight='balanced'
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=96),
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Fit the model
grid_search.fit(X_train, y_train)
best_clf = grid_search.best_estimator_

# Evaluate
y_pred = best_clf.predict(X_test)
print("Best Parameters:", grid_search.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Optional: show feature importances
importances = best_clf.feature_importances_
for feature, score in sorted(zip(features_list[1:], importances), key=lambda x: x[1], reverse=True):
    print(f"{feature}: {score:.4f}")

# Save the best model and metadata
with open("my_classifier.pkl", "wb") as clf_out:
    pickle.dump(best_clf, clf_out)

with open("my_dataset.pkl", "wb") as dataset_out:
    pickle.dump(data_dict, dataset_out)

with open("my_feature_list.pkl", "wb") as features_out:
    pickle.dump(features_list, features_out)

