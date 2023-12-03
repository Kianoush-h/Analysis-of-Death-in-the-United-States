
"""
@author: Kianoush 

GitHUb: https://github.com/Kianoush-h
YouTube: https://www.youtube.com/channel/UCvf9_53f6n3YjNEA4NxAkJA
LinkedIn: https://www.linkedin.com/in/kianoush-haratiannejadi/

Email: haratiank2@gmail.com

"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Load all CSV files into a list of DataFrames
file_path = 'data/2014_data.csv'

encoding = "utf-8"
# encoding = "ISO-8859-1"
f = open('data/2014_codes.json')
description = json.load(f)

raw_data = pd.read_csv(file_path, encoding = encoding)

print(raw_data.shape)

# Display the first few rows
head = raw_data.head(5)

# Display the last few rows
raw_data.tail(5)

# Basic statistics of numerical columns
temp = raw_data.describe()

categorical_stats = raw_data.describe(include=['object'])


raw_data.info()



# =============================================================================
# PART 0 - finding interesting columns 
#
# month_of_death
# day_of_week_of_death
# manner_of_death
# sex
# detail_age
# race
#
# marital_status
# place_of_injury_for_causes_w00_y34_except_y06_and_y07_
# =============================================================================



description["marital_status"]
raw_data['race']


# =============================================================================
# Data Visualization
# =============================================================================



# =============================================================================
# PART 1 - Death rate by months and days of a week
# =============================================================================

dec_map = {int(key): value for key, value in description["day_of_week_of_death"].items()}
raw_data['day_name'] = raw_data['day_of_week_of_death'].map(dec_map)

# Sort the days of the week
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'Unknown']
raw_data['day_name'] = pd.Categorical(raw_data['day_name'], categories=days_order, ordered=True)


dec_map = {int(key): value for key, value in description["month_of_death"].items()}
raw_data['month_of_death_name'] = raw_data['month_of_death'].map(dec_map)

sns.countplot(x=raw_data['month_of_death_name'], hue=raw_data['day_name'])



# =============================================================================
# PART 2 - Death rate by manner_of_death
# =============================================================================

raw_data['manner_of_death'].isnull().sum()

dec_map = {"9" if key == "Blank" else key: value for key, value in description["manner_of_death"].items()}
dec_map = {int(key): value for key, value in dec_map.items()}

raw_data['manner_of_death'] = raw_data['manner_of_death'].fillna(9)
raw_data['manner_of_death_name'] = raw_data['manner_of_death'].map(dec_map)



# Calculate the ticket counts for each combined location
if len(set(raw_data['manner_of_death_name'])) < 10:
    top_x = len(set(raw_data['manner_of_death_name']))
else:
    top_x = 10
status = raw_data['manner_of_death_name'].value_counts().nlargest(top_x)

# Use a color palette from seaborn
colors = sns.color_palette("pastel")

# Explode and Shadow
explode = (0.1,) + tuple(0 for i in range( top_x-1))  # Highlight the first slice
shadow = True

# Plotting a pie chart for the top 10 
plt.figure(figsize=(10, 10))
plt.pie(status, labels=status.index, autopct='%1.1f%%', startangle=140, colors=colors, explode=explode, shadow=shadow)

# Title and labels
plt.title(f'Top {top_x} Status', fontsize=16)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

# Legend
# plt.legend(location_counts.index, title='Locations', loc='upper right')

plt.show()



# =============================================================================
# PART 2 - Death rate by manner_of_death and gender
# =============================================================================

sns.countplot(x=raw_data['manner_of_death_name'], hue=raw_data['sex'])



# =============================================================================
# PART 3 - Death rate by manner_of_death and age
# =============================================================================

# Use a color palette from seaborn
sns.set_palette("pastel")

plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")  # Add grid lines

raw_data.groupby("manner_of_death_name")["detail_age"].mean().plot(kind='bar')

# Labels and titles
plt.title('Death by manner_of_death and Age', fontsize=16)
plt.xlabel('manner_of_death', fontsize=14)
plt.ylabel('Age', fontsize=14)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Data labels
for index, value in enumerate(raw_data.groupby('manner_of_death_name')['detail_age'].mean()):
    plt.text(index, value + 1, str(int(value)), ha='center', va='bottom')

plt.show()





# =============================================================================
# PART 6
# =============================================================================



# Calculate correlation matrix
correlation_matrix = raw_data.corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

















