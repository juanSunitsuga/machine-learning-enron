#!/usr/bin/python3
import os
import joblib
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath("C:/Users/juank/Documents/KULIAH/Semester 4/Machine Learning/Week 6/tools/"))
from feature_format import featureFormat

# Load data
data_dict = joblib.load(open("C:\\Users\\juank\\Documents\\KULIAH\\Semester 4\\Machine Learning\\Tugas Besar\\final_project\\final_project_dataset.pkl", "rb"))

email_features = [
    'to_messages', 'from_poi_to_this_person', 'from_messages',
    'from_this_person_to_poi', 'shared_receipt_with_poi'
]

def find_outliers_iqr(data_list):
    q1 = np.percentile(data_list, 25)
    q3 = np.percentile(data_list, 75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    return lower_bound, upper_bound

# Detect outliers for each email feature using IQR
outlier_dict = {feature: [] for feature in email_features}
for feature in email_features:
    values = [float(features[feature]) for name, features in data_dict.items()
              if features[feature] != 'NaN']
    lower, upper = find_outliers_iqr(values)
    for name, features in data_dict.items():
        if features[feature] != 'NaN':
            val = float(features[feature])
            if val < lower or val > upper:
                outlier_dict[feature].append((name, val))

print("\nTop Outliers in Email Features:")
for feature in email_features:
    print(f"\nTop outliers for {feature}:")
    for name, val in sorted(outlier_dict[feature], key=lambda x: abs(x[1]), reverse=True)[:5]:
        print(f"{name}: {val}")

# Find individuals who are outliers in multiple email features
multi_email_outliers = {name: 0 for name in data_dict.keys()}
for feature in email_features:
    for name, _ in outlier_dict[feature]:
        multi_email_outliers[name] += 1

print("\nIndividuals who are outliers in multiple email features:")
for name, count in sorted(multi_email_outliers.items(), key=lambda x: x[1], reverse=True):
    if count >= 2:
        print(f"{name}: Outlier in {count} email features")

# Plot scatter plots for each pair of email features, highlighting outliers
plt.figure(figsize=(18, 12))
plot_num = 1
for i in range(len(email_features)):
    for j in range(i+1, len(email_features)):
        plt.subplot(3, 4, plot_num)
        x_feat = email_features[i]
        y_feat = email_features[j]
        xs = []
        ys = []
        for name, features in data_dict.items():
            x = features[x_feat]
            y = features[y_feat]
            if x != 'NaN' and y != 'NaN':
                xs.append(float(x))
                ys.append(float(y))
        plt.scatter(xs, ys, color='blue', alpha=0.6, label='Normal')
        # Highlight outliers in red
        outlier_names = set([n for n, _ in outlier_dict[x_feat]] + [n for n, _ in outlier_dict[y_feat]])
        for name in outlier_names:
            fx = data_dict[name][x_feat]
            fy = data_dict[name][y_feat]
            if fx != 'NaN' and fy != 'NaN':
                plt.scatter(float(fx), float(fy), color='red', s=60, edgecolor='k', label='Outlier')
                plt.annotate(name, (float(fx), float(fy)), fontsize=7, color='red')
        plt.xlabel(x_feat)
        plt.ylabel(y_feat)
        plt.tight_layout()
        plot_num += 1

plt.suptitle("Email Feature Outliers (Red = Outlier)", fontsize=16)
plt.savefig("email_feature_outliers.png", dpi=300, bbox_inches='tight')
plt.show()