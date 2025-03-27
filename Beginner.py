# 1. Basic Python Programming


print("Hello, World!")


name = "Sannidhya"
age = 19
cgpa = 9.19
abc = False 
print (name,age,cgpa,abc)


sum = 10+20
product = 5 * 3
remainder = 15 % 4
division = 18/3 
print (sum,product,remainder,division)


# 2. Control Structures 


num = -21
if num > 0:
    print("Positive number")
elif num < 0:
    print("Negative number")
else:
    print("Zero")


i=0
s=0
for i in range(5):
    s=s+i
print (s)



count = 0
while (count < 5):
    print(count)
    count += 1
print ("final value of count =", count)


# 3. Numpy Essentials

import numpy as np
arr1=np.array([1,2,3,4,5])
arr2=np.random.rand(3)
print (arr1 , "\n" ,  arr2)

sum1=arr1 + 5
prod=arr1 * 2
print (sum,"\n",prod)


mean=np.mean(arr1)
median = np.median(arr1)
sum2=np.sum(arr1)
print ("Mean =",mean,"\n Median=",median , "\n Sum=",sum2)


# 4. Data Structures with Pandas

import pandas as pd
data = { 'Name': ['Sanu', 'Sakchham', 'Anikait'],'Age': [19, 19, 18],'City': ['Manipal', 'Kharagpur', 'Ranchi']}
df = pd.DataFrame(data)
print (df)

agedf= df[df['Age']>18]
sorting= df.sort_values('Name')
indexing= df.set_index('Name')
print(agedf,"\n",sorting,"\n",indexing)


# 5. Basic Data Analysis

mean_age = df['Age'].mean()
median_age = df['Age'].median()
std_age = df['Age'].std()
print("Mean Age=",mean_age, "\n Median Age=", median_age,"\n Standard deviation age=",std_age)


ages = df['Age'].values
np_mean = np.mean(ages)
np_std = np.std(ages)
print("Ages:",ages,"Mean=",np_mean,"Standard deviation=",np_std)


# 6. Matplotlib Basics

import matplotlib.pyplot as plt
a1= [1, 2, 3, 4, 5]
b1 = [1, 2, 3, 4, 5]
plt.plot(a1, b1)
plt.title("Simple Line Chart")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

x = np.random.rand(50)
y = np.random.rand(50)
plt.scatter(x, y)
plt.title("Scatter Plot")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.show()


# 7. Data Visualisation with Seaborn
import seaborn as sns 

tips = sns.load_dataset('tips')


plt.figure(figsize=(8, 6))
sns.boxplot(x='day', y='total_bill', data=tips)
plt.title('Box Plot of Total Bill by Day')
plt.show()


plt.figure(figsize=(8, 6))
sns.histplot(tips['total_bill'], kde=True)
plt.title('Histogram with Density Plot')
plt.show()


# 8. Time Series Data

dates = pd.date_range('2023-01-01', periods=100)
values = np.random.randn(100).cumsum()
ts = pd.Series(values, index=dates)

plt.figure(figsize=(10, 6))
ts.plot()
plt.title('Random Walk Time Series')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True)
plt.show()


# 9. Correlation Analysis

plt.figure(figsize=(8, 6))
corr = tips.corr(numeric_only=True)
sns.heatmap(corr, annot=True)
plt.title('Correlation Analysis')
plt.show()


# 10. Data Aggregation

group = tips.groupby('day')['total_bill'].mean()
plt.figure(figsize=(8, 6))
group.plot(kind='bar')
plt.title('Average Total Bill by Day')
plt.ylabel('Amount')
plt.show()


# 11. Data Cleaning

data1 = {'A': [1, 2, np.nan, 4], 'B': [5, np.nan, np.nan, 8], 'C': [9, 10, 11, 12]}
df = pd.DataFrame(data1)
print("Original DataFrame:")
print(df)
df_dropped = df.dropna(how='all')
df_filled = df.fillna(df.mean())
print("\n After dropping all- NA rows:")
print(df_dropped)
print("\nAfter filling with mean:")
print(df_filled)


# 12. Combining Plots


fig, axes = plt.subplots(1, 3, figsize=(18, 5))


axes[0].plot([1, 2, 3, 4], [1, 4, 9, 16])
axes[0].set_title('Line Plot')


axes[1].bar(['A', 'B', 'C'], [3, 7, 2])
axes[1].set_title('Bar Plot')


axes[2].scatter(np.random.rand(20), np.random.rand(20))
axes[2].set_title('Scatter Plot')

plt.tight_layout()
plt.show()

# 13. Custom Visualisation 

x1 = np.linspace(0, 10, 100)
y1 = np.sin(x1)

plt.figure(figsize=(8, 6))
plt.plot(x1, y1, color='purple', linestyle='--', linewidth=2, marker='o', markersize=5, label='sin(x)')
plt.title('Customized Sine Wave', fontsize=14)
plt.xlabel('X-axis', fontsize=12)
plt.ylabel('Y-axis', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle=':', alpha=0.7)
plt.show()


# 14. Exploratory Data Analysis (EDA)

iris = sns.load_dataset('iris')
print("First 5 rows:")
print(iris.head())
print("\n Summary statistics:")
print(iris.describe())
print("\n Species count:")
print(iris['species'].value_counts())
sns.pairplot(iris)
plt.show()


# 15. Mini Project 

sns.set_style("whitegrid")

data3 = {'Name': ['Sanu', 'Sakchham', 'Anikait', 'Jayesh', 'Akshay'],'Math_Score': [90, 100, 90, 92, 70],'Science_Score': [78, 88, 92, 65, 70],'Hours_Studied': [10, 15, 20, 8, 5]}
df3 = pd.DataFrame(data3)
print("Original Data:")
print(df3.head())

print("\n Missing values:")
print(df3.isnull().sum())

df3['Total_Score'] = df3['Math_Score'] + df3['Science_Score']
print("\n Data with Total Score:")
print(df3)

print("\n Summary Statistics:")
print(df3.describe())


math_max = df3[df3['Math_Score'] == df3['Math_Score'].max()]
print("\n Top Math Student:")
print(math_max[['Name', 'Math_Score']])

science_max = df3[df3['Science_Score'] == df3['Science_Score'].max()]
print("\n Top Science Student:")
print(math_max[['Name', 'Science_Score']])


plt.figure(figsize=(6, 4))
df3.plot(x='Name', y=['Math_Score', 'Science_Score'], kind='bar')
plt.title("Student Scores by Subject")
plt.ylabel("Score")
plt.show()

print("\n Key Insights:")
print(f"- Highest Math Score: {df3['Math_Score'].max()} (by {math_max['Name'].values[0]})")
print(f"- Average Science Score: {df3['Science_Score'].mean():.1f}")
print("- Students who studied more hours generally scored higher.")

