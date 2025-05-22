#!/usr/bin/python3
import os
import joblib
import sys
import matplotlib.pyplot
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy
from pprint import pprint
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.abspath("C:/Users/ACER/Documents/KULIAH/Semester 4/Machine Learning/Week 6/tools/"))
from feature_format import featureFormat, targetFeatureSplit

### read in data dictionary, convert to numpy array
data_dict = joblib.load(open("C:\\Users\\ACER\\Documents\\KULIAH\\Semester 4\\Machine Learning\\Week 6\\Mini_Projects\\final_project\\final_project_dataset.pkl", "rb"))

financialFeatures = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
financialData = featureFormat(data_dict, financialFeatures)


### your code below
# 1. Better visualization with outliers labeled
plt.figure(figsize=(12, 8))
for point in financialData:
    salary = point[0]
    bonus = point[4]  # Using index 4 for bonus based on financialFeatures list
    plt.scatter(salary, bonus, color='blue', alpha=0.6)

# Annotate the extreme outliers directly on the plot
for name, features in data_dict.items():
    if features['salary'] != 'NaN' and features['bonus'] != 'NaN':
        if features['salary'] > 1000000 and features['bonus'] > 5000000:
            plt.annotate(name, 
                        xy=(features['salary'], features['bonus']),
                        xytext=(10, 5),
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='red'))

plt.xlabel('Salary ($)')
plt.ylabel('Bonus ($)')
plt.title('Enron Salary vs. Bonus with Outliers Highlighted')
plt.grid(True, alpha=0.3)
plt.tight_layout()
# Save figure before showing it
plt.savefig('salary_vs_bonus_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Create new derived features that might help identify outliers
# Let's create a feature for bonus-to-salary ratio and total compensation
my_dataset = data_dict.copy()

for name, features in my_dataset.items():
    # Create bonus-to-salary ratio
    if features['salary'] != 'NaN' and features['bonus'] != 'NaN' and float(features['salary']) > 0:
        features['bonus_to_salary_ratio'] = float(features['bonus']) / float(features['salary'])
    else:
        features['bonus_to_salary_ratio'] = 'NaN'
    
    # Create total_compensation feature (salary + bonus + stock value)
    has_salary = features['salary'] != 'NaN'
    has_bonus = features['bonus'] != 'NaN'
    has_stock = features['total_stock_value'] != 'NaN'
    
    total_comp = 0
    if has_salary:
        total_comp += float(features['salary'])
    if has_bonus:
        total_comp += float(features['bonus'])
    if has_stock:
        total_comp += float(features['total_stock_value'])
    
    if has_salary or has_bonus or has_stock:
        features['total_compensation'] = total_comp
    else:
        features['total_compensation'] = 'NaN'

# Add new features to your feature list
my_feature_list = financialFeatures + ['bonus_to_salary_ratio', 'total_compensation']

# 3. Statistical outlier detection using the new features
bonus_ratio_values = [v['bonus_to_salary_ratio'] for k, v in my_dataset.items() 
                     if v['bonus_to_salary_ratio'] != 'NaN']
total_comp_values = [v['total_compensation'] for k, v in my_dataset.items() 
                    if v['total_compensation'] != 'NaN']

# Using IQR method to identify outliers in the new features
def find_outliers_iqr(data_list):
    """Find outliers using the IQR method"""
    q1 = numpy.percentile(data_list, 25)
    q3 = numpy.percentile(data_list, 75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    
    outliers = [(i, x) for i, x in enumerate(data_list) if x < lower_bound or x > upper_bound]
    return outliers, lower_bound, upper_bound

# Find outliers in bonus-to-salary ratio
ratio_outliers, ratio_lower, ratio_upper = find_outliers_iqr(bonus_ratio_values)

# Find outliers in total compensation
comp_outliers, comp_lower, comp_upper = find_outliers_iqr(total_comp_values)

# Print information about the outliers in the new features
print("\nOutliers in Bonus-to-Salary Ratio:")
ratio_outlier_names = []
for name, features in my_dataset.items():
    if features['bonus_to_salary_ratio'] != 'NaN':
        if features['bonus_to_salary_ratio'] > ratio_upper:
            ratio_outlier_names.append((name, features['bonus_to_salary_ratio']))

for name, ratio in sorted(ratio_outlier_names, key=lambda x: x[1], reverse=True)[:5]:
    print(f"{name}: Bonus/Salary Ratio = {ratio:.2f}")

print("\nOutliers in Total Compensation:")
comp_outlier_names = []
for name, features in my_dataset.items():
    if features['total_compensation'] != 'NaN':
        if features['total_compensation'] > comp_upper:
            comp_outlier_names.append((name, features['total_compensation']))

for name, comp in sorted(comp_outlier_names, key=lambda x: x[1], reverse=True)[:5]:
    print(f"{name}: Total Compensation = ${comp:,.2f}")

# 4. Visualize the new features to spot potential outliers
plt.figure(figsize=(15, 6))

# Plot bonus-to-salary ratio
plt.subplot(1, 2, 1)
plt.boxplot(bonus_ratio_values)
plt.title('Bonus-to-Salary Ratio Distribution')
plt.ylabel('Ratio Value')
plt.grid(True, alpha=0.3)

# Plot total compensation
plt.subplot(1, 2, 2)
plt.boxplot(total_comp_values)
plt.title('Total Compensation Distribution')
plt.ylabel('Amount ($)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('derived_features_boxplots.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. Create a function to remove identified outliers
def remove_outliers(dataset, outlier_list):
    """Remove specified outliers from the dataset"""
    cleaned_data = dataset.copy()
    for outlier in outlier_list:
        if outlier in cleaned_data:
            del cleaned_data[outlier]
    return cleaned_data

# Example: Remove the most extreme outliers (those with both high salary and high bonus)
extreme_outliers = name  # using the list you already created
cleaned_dataset = remove_outliers(my_dataset, extreme_outliers)

print(f"\nOriginal dataset size: {len(my_dataset)}")
print(f"Cleaned dataset size: {len(cleaned_dataset)}")
print(f"Removed {len(extreme_outliers)} extreme outliers: {extreme_outliers}")

# Define the specific features you want to analyze
selected_features = ['salary', 'total_payments', 'bonus', 'long_term_incentive']

# Extract these features from the dataset
feature_data = featureFormat(data_dict, selected_features)

# 1. Create a pairplot matrix to visualize relationships between these features
plt.figure(figsize=(15, 15))

# Create 2x2 grid of scatter plots for all feature combinations
feature_names = {0: 'Salary', 1: 'Total Payments', 2: 'Bonus', 3: 'Long Term Incentive'}
for i in range(len(selected_features)):
    for j in range(len(selected_features)):
        if i != j:  # Don't plot same feature against itself
            plt.subplot(4, 4, i*4 + j + 1)
            for point in feature_data:
                plt.scatter(point[j], point[i], color='blue', alpha=0.5)
                
            # Label outliers on specific combinations if desired
            if (i == 0 and j == 2) or (i == 2 and j == 0):  # Salary vs Bonus or vice versa
                for name, features in data_dict.items():
                    if (features[selected_features[j]] != 'NaN' and 
                        features[selected_features[i]] != 'NaN'):
                        # Set appropriate thresholds for outliers
                        if features[selected_features[j]] > 1000000 and features[selected_features[i]] > 1000000:
                            plt.annotate(name, 
                                        xy=(features[selected_features[j]], features[selected_features[i]]),
                                        xytext=(5, 5),
                                        textcoords='offset points',
                                        fontsize=8,
                                        arrowprops=dict(arrowstyle='->', color='red'))
            
            plt.xlabel(feature_names[j])
            plt.ylabel(feature_names[i])
            plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle('Relationships Between Selected Financial Features', y=1.02, fontsize=16)
plt.savefig('feature_relationships_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Identify outliers using boxplots for each feature
plt.figure(figsize=(15, 6))

# Create data lists for each feature, filtering out 'NaN' values
feature_values = []
for i, feature in enumerate(selected_features):
    values = [features[feature] for name, features in data_dict.items() 
             if features[feature] != 'NaN']
    feature_values.append(values)

# Create boxplots
for i, feature in enumerate(selected_features):
    plt.subplot(1, 4, i+1)
    plt.boxplot(feature_values[i])
    plt.title(feature_names[i])
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('selected_features_boxplots.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Create derived features specific to these 4 features
my_dataset = data_dict.copy()

for name, features in my_dataset.items():
    # Ratio of bonus to total payments
    if features['bonus'] != 'NaN' and features['total_payments'] != 'NaN' and float(features['total_payments']) > 0:
        features['bonus_to_total_ratio'] = float(features['bonus']) / float(features['total_payments'])
    else:
        features['bonus_to_total_ratio'] = 'NaN'
    
    # Ratio of long term incentive to salary
    if features['long_term_incentive'] != 'NaN' and features['salary'] != 'NaN' and float(features['salary']) > 0:
        features['lti_to_salary_ratio'] = float(features['long_term_incentive']) / float(features['salary'])
    else:
        features['lti_to_salary_ratio'] = 'NaN'

# 4. Print top outliers for each of the 4 features
print("\nTop Outliers in Selected Features:")
for i, feature in enumerate(selected_features):
    print(f"\nTop 5 outliers for {feature_names[i]}:")
    outlier_list = []
    for name, features in data_dict.items():
        if features[feature] != 'NaN':
            outlier_list.append((name, float(features[feature])))
    
    # Sort and print top 5 
    for name, value in sorted(outlier_list, key=lambda x: x[1], reverse=True)[:5]:
        print(f"{name}: ${value:,.2f}")

# 5. Find multivariate outliers (people who are outliers in multiple features)
multi_outliers = {}
for name in data_dict.keys():
    multi_outliers[name] = 0

# Count how many features each person is an outlier in
for feature in selected_features:
    # Get values for this feature
    values = [float(features[feature]) for name, features in data_dict.items() 
             if features[feature] != 'NaN']
    
    # Calculate outlier threshold (using simple percentile method)
    threshold = numpy.percentile(values, 95)  # Top 5%
    
    # Count outliers
    for name, features in data_dict.items():
        if features[feature] != 'NaN' and float(features[feature]) > threshold:
            multi_outliers[name] += 1

# Print people who are outliers in multiple features
print("\nIndividuals who are outliers in multiple features:")
for name, count in sorted(multi_outliers.items(), key=lambda x: x[1], reverse=True):
    if count >= 2:  # At least 2 features
        print(f"{name}: Outlier in {count} of the 4 selected features")