import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Dataset
data = pd.read_csv("train.csv")

# Clean Missing Values
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
data['Fare'] = data['Fare'].fillna(data['Fare'].median())

sns.set_style("whitegrid")

# 1. Survival by Gender
plt.figure(figsize=(7,5))
sns.countplot(x='Sex', hue='Survived', data=data, palette="coolwarm")
plt.title("Survival by Gender", fontsize=14, fontweight="bold")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.savefig("survival_by_gender.png")
plt.show()

# 2. Survival by Passenger Class
plt.figure(figsize=(7,5))
sns.countplot(x='Pclass', hue='Survived', data=data, palette="Set2")
plt.title("Survival by Passenger Class", fontsize=14, fontweight="bold")
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.savefig("survival_by_class.png")
plt.show()

# 3. Age Distribution
plt.figure(figsize=(7,5))
sns.histplot(data['Age'], bins=30, kde=True, color="purple")
plt.title("Age Distribution of Passengers", fontsize=14, fontweight="bold")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.savefig("age_distribution.png")
plt.show()

# 4. Correlation Heatmap (only numeric columns)
plt.figure(figsize=(8,6))
sns.heatmap(data.select_dtypes(include='number').corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap of Numeric Features", fontsize=14, fontweight="bold")
plt.savefig("correlation_heatmap.png")
plt.show()

print("4 Charts Saved: survival_by_gender.png, survival_by_class.png, age_distribution.png, correlation_heatmap.png")
