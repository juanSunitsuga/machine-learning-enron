import pickle
import joblib
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# Load the dataset
data_dict = joblib.load(open("D:\Perkuliahan\File Kuliah Semester 4\ML\Tubes\machine-learning-enron\\final_project_dataset.pkl", "rb"))

# Prepare features and labels
features_list = ['poi', 'salary']
names = np.array(list(data_dict.keys()))
data = []
labels = []
for name in names:
    person = data_dict[name]
    salary = person['salary']
    poi = person['poi']
    # Skip if salary is 'NaN'
    if salary == 'NaN':
        continue
    data.append([float(salary)])
    labels.append(int(poi))
names = names[[person['salary'] != 'NaN' for person in data_dict.values()]]

X = np.array(data)
y = np.array(labels)

# Split data and keep indices
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, np.arange(len(X)), test_size=0.3, random_state=42, stratify=y
)

# Fit and predict
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Show right and wrong predictions
print("\nCorrectly classified (right):")
for i, pred in enumerate(y_pred):
    if pred == y_test[i]:
        print(f"  {names[idx_test[i]]} (True label: {y_test[i]}, Predicted: {pred})")

print("\nIncorrectly classified (wrong):")
for i, pred in enumerate(y_pred):
    if pred != y_test[i]:
        print(f"  {names[idx_test[i]]} (True label: {y_test[i]}, Predicted: {pred})")