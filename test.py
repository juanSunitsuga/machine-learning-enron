import pickle
import joblib
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# Load the dataset
data_dict = joblib.load(open("C:\\Users\\juank\\Documents\\KULIAH\\Semester 4\\Machine Learning\\Tugas Besar\\final_project\\final_project_dataset_cleaned.pkl", "rb"))

# Prepare features and labels
features_list = ['poi', 'salary']
data = []
labels = []
names = []

for name, person in data_dict.items():
    salary = person['salary']
    poi = person['poi']
    # Skip if salary is 'NaN'
    if salary == 'NaN':
        continue
    data.append([float(salary)])
    labels.append(int(poi))
    names.append(name)

X = np.array(data)
y = np.array(labels)
names = np.array(names)

print(f"Total people with valid salary data: {len(names)}")
print(f"POIs: {sum(y)}, Non-POIs: {len(y) - sum(y)}")

# Split data and keep indices
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, np.arange(len(X)), test_size=0.3, random_state=42, stratify=y
)

# Fit and predict
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = sum(y_pred == y_test) / len(y_test)
print(f"\nAccuracy: {accuracy:.3f}")

# Show right and wrong predictions
print(f"\nCorrectly classified ({sum(y_pred == y_test)} people):")
for i, pred in enumerate(y_pred):
    if pred == y_test[i]:
        poi_status = "POI" if y_test[i] == 1 else "Non-POI"
        print(f"  {names[idx_test[i]]} - {poi_status}")

print(f"\nIncorrectly classified ({sum(y_pred != y_test)} people):")
for i, pred in enumerate(y_pred):
    if pred != y_test[i]:
        true_status = "POI" if y_test[i] == 1 else "Non-POI"
        pred_status = "POI" if pred == 1 else "Non-POI"
        print(f"  {names[idx_test[i]]} - True: {true_status}, Predicted: {pred_status}")

# Show detailed breakdown
print(f"\nDetailed Results:")
print(f"True Positives (correctly identified POIs): {sum((y_test == 1) & (y_pred == 1))}")
print(f"True Negatives (correctly identified Non-POIs): {sum((y_test == 0) & (y_pred == 0))}")
print(f"False Positives (incorrectly identified as POIs): {sum((y_test == 0) & (y_pred == 1))}")
print(f"False Negatives (missed POIs): {sum((y_test == 1) & (y_pred == 0))}")