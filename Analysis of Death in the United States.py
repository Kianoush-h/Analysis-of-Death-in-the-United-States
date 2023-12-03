
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

# raw_data.dropna(inplace=True)

# raw_data['ORDERDATE'] = pd.to_datetime(raw_data['ORDERDATE'], infer_datetime_format=True)



# raw_data['year'] = raw_data['date_of_infraction'].dt.year
# raw_data['month'] = raw_data['date_of_infraction'].dt.month
# raw_data['day_of_week'] = raw_data['ORDERDATE'].dt.dayofweek


# =============================================================================
# PART 0 - finding interesting columns 
#
# month_of_death
# day_of_week_of_death
# manner_of_death
#
# marital_status
# detail_age_type
# education_2003_revision
# marital_status
# place_of_injury_for_causes_w00_y34_except_y06_and_y07_
# race
# sex
# =============================================================================



description["manner_of_death"]
raw_data['detail_age_type']


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
# PART 2 - Death rate by manner_of_death and sex
# =============================================================================



sns.countplot(x=raw_data['manner_of_death_name'], hue=raw_data['sex'])











# Use a color palette from seaborn
sns.set_palette("pastel")

# Bar plot for Temporal Trends by Day of the Week with sorted day names
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")  # Add grid lines
df = raw_data.groupby('manner_of_death_name')['sex'].count().sort_index().plot(kind='bar')

# Labels and titles
plt.title('Death Rate by manner_of_death_name', fontsize=16)
plt.xlabel('manner_of_death_name', fontsize=14)
plt.ylabel('Number of Death', fontsize=14)

# Legend
# plt.legend(['Race'], loc='upper right')

# Data labels
for index, value in enumerate(raw_data.groupby('manner_of_death_name')['sex'].count().sort_index()):
    plt.text(index, value + 1, str(value), ha='center', va='bottom')

plt.grid(True)
plt.show()





# =============================================================================
# PART 2 - Death rate by race
# =============================================================================




race_map = {int(key): value for key, value in description["race"].items()}
raw_data['race_description'] = raw_data['race'].map(race_map)

# Use a color palette from seaborn
sns.set_palette("pastel")

# Bar plot for Temporal Trends by Day of the Week with sorted day names
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")  # Add grid lines
df = raw_data.groupby('race_description')['sex'].count().sort_index().plot(kind='bar')

# Labels and titles
plt.title('Death Rate by Race', fontsize=16)
plt.xlabel('Race', fontsize=14)
plt.ylabel('Number of Death', fontsize=14)

# Legend
# plt.legend(['Race'], loc='upper right')

# Data labels
for index, value in enumerate(raw_data.groupby('race_description')['sex'].count().sort_index()):
    plt.text(index, value + 1, str(value), ha='center', va='bottom')

plt.grid(True)
plt.show()







# Use a color palette from seaborn
sns.set_palette("viridis")

# Create the first subplot (top)
plt.figure(figsize=(12, 12))

plt.subplot(2, 1, 1)

# Line plot for Temporal Trends by Months with sorted month names
sns.lineplot(data=raw_data.groupby('month_name')['SALES'].sum().sort_index(), marker='o', linestyle='-')

# Labels and titles for the first subplot
plt.title('SALES Over the Months', fontsize=16)
plt.xlabel('Month', fontsize=14)
plt.ylabel('SALES', fontsize=14)
plt.grid(True)

# Data labels for the first subplot
for index, value in enumerate(raw_data.groupby('month_name')['SALES'].sum().sort_index()):
    plt.text(index, value + 1, str(value), ha='center', va='bottom')


# Graph 2
# Use a color palette from seaborn
sns.set_palette("viridis")

# Create the second subplot (bottom)
plt.subplot(2, 1, 2)

# Line plot for Temporal Trends by Months with sorted month names
sns.lineplot(data=raw_data.groupby('month_name')['QUANTITYORDERED'].sum().sort_index(), marker='o', linestyle='-')

# Labels and titles for the second subplot
plt.title('QUANTITY ORDERED Over the Months', fontsize=16)
plt.xlabel('Month', fontsize=14)
plt.ylabel('QUANTITY ORDERED', fontsize=14)
plt.grid(True)

# Data labels for the second subplot
for index, value in enumerate(raw_data.groupby('month_name')['QUANTITYORDERED'].sum().sort_index()):
    plt.text(index, value + 1, str(value), ha='center', va='bottom')

# Adjust layout to prevent clipping of titles and labels
plt.tight_layout()

# Show the plot
plt.show()




# =============================================================================
# PART 3
# =============================================================================


# Adjust the number of top descriptions to show (e.g., top 10)
top_n = 10
top_descriptions = raw_data['PRODUCTCODE'].value_counts().nlargest(top_n).index

# Use a color palette from seaborn
sns.set_palette("pastel")

plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")  # Add grid lines

# Filter data for top PRODUCTCODE
filtered_data = raw_data[raw_data['PRODUCTCODE'].isin(top_descriptions)]

# Bar plot for sales by Top PRODUCTCODE
filtered_data.groupby('PRODUCTCODE')['SALES'].sum().plot(kind='bar')

# Labels and titles
plt.title('SALES by Top PRODUCTCODE', fontsize=16)
plt.xlabel('PRODUCTCODE', fontsize=14)
plt.ylabel('SALES', fontsize=14)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Data labels
for index, value in enumerate(filtered_data.groupby('PRODUCTCODE')['SALES'].sum()):
    plt.text(index, value + 1, str(value), ha='center', va='bottom')

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


# =============================================================================
# PART 7
# =============================================================================



# Example: Time series plot of parking tickets count by date
plt.figure(figsize=(16, 8))
raw_data['ORDERDATE'].value_counts().sort_index().plot(kind='line')
plt.title('SALES Over Time')
plt.xlabel('DATE')
plt.ylabel('SALES')
plt.show()



















