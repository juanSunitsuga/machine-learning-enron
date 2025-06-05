#!/usr/bin/python3
import os
import joblib
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

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
        x_feat = email_features[i]
        y_feat = email_features[j]
        xs = []
        ys = []
        names = []
        for name, features in data_dict.items():
            x = features[x_feat]
            y = features[y_feat]
            if x != 'NaN' and y != 'NaN':
                xs.append(float(x))
                ys.append(float(y))
                names.append(name)
        xs_np = np.array(xs).reshape(-1, 1)
        ys_np = np.array(ys)
        # Fit regression
        reg = LinearRegression().fit(xs_np, ys_np)
        preds = reg.predict(xs_np)
        residuals = np.abs(ys_np - preds)
        # Find top 5 outliers by residual
        outlier_indices = residuals.argsort()[-5:][::-1]
        outlier_names = [names[idx] for idx in outlier_indices]
        plt.subplot(3, 4, plot_num)
        plt.scatter(xs, ys, color='blue', alpha=0.6, label='Normal')
        # Highlight outliers in red
        for idx in outlier_indices:
            plt.scatter(xs[idx], ys[idx], color='red', s=60, edgecolor='k', label='Outlier')
            plt.annotate(names[idx], (xs[idx], ys[idx]), fontsize=7, color='red')
        # Plot regression line
        x_line = np.linspace(min(xs), max(xs), 100).reshape(-1, 1)
        y_line = reg.predict(x_line)
        plt.plot(x_line, y_line, color='green', linestyle='--', linewidth=1)
        plt.xlabel(x_feat)
        plt.ylabel(y_feat)
        plt.tight_layout()
        plot_num += 1

plt.suptitle("Email Feature Outliers by Regression Residuals (Red = Outlier)", fontsize=16)
plt.savefig("email_feature_outliers_regression.png", dpi=300, bbox_inches='tight')
plt.show()