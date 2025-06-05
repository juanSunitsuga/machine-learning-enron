import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load pickled data
with open("my_classifier.pkl", "rb") as clf_in:
    clf = pickle.load(clf_in)

with open("my_dataset.pkl", "rb") as dataset_in:
    data_dict = pickle.load(dataset_in)

with open("my_feature_list.pkl", "rb") as features_in:
    features_list = pickle.load(features_in)

# Prepare feature matrix and labels
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

# Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []
precisions = []
recalls = []
f1_scores = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracies.append(accuracy_score(y_test, y_pred))
    precisions.append(precision_score(y_test, y_pred, zero_division=0))
    recalls.append(recall_score(y_test, y_pred, zero_division=0))
    f1_scores.append(f1_score(y_test, y_pred, zero_division=0))

    print(f"Fold {fold}: Accuracy={accuracies[-1]:.4f}, Precision={precisions[-1]:.4f}, Recall={recalls[-1]:.4f}, F1={f1_scores[-1]:.4f}")

# Average metrics
print("\n=== Cross-Validation Summary ===")
print(f"Average Accuracy:  {np.mean(accuracies):.4f}")
print(f"Average Precision: {np.mean(precisions):.4f}")
print(f"Average Recall:    {np.mean(recalls):.4f}")
print(f"Average F1 Score:  {np.mean(f1_scores):.4f}")
