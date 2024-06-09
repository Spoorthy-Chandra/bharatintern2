import pandas as pd
​
train_data = pd.read_csv('/kaggle/input/titanic-survivor-classification/train.csv')
test_data = pd.read_csv('/kaggle/input/titanic-survivor-classification/test.csv')
​
train_data.head()

train_data.info()


train_data.describe()


train_data.isnull().sum()


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Survived', data=train_data)
plt.show()

import pandas as pd
​
​
train_data = pd.read_csv('/kaggle/input/titanic-survivor-classification/train.csv')
test_data = pd.read_csv('/kaggle/input/titanic-survivor-classification/test.csv')
​
​
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())
​
​
embarked_mode = train_data['Embarked'].mode()[0]
train_data['Embarked'] = train_data['Embarked'].fillna(embarked_mode)
​
​
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())
​
​
train_data = pd.get_dummies(train_data, columns=['Sex', 'Embarked'])
test_data = pd.get_dummies(test_data, columns=['Sex', 'Embarked'])


train_data, test_data = train_data.align(test_data, join='left', axis=1)


train_data = train_data.drop(['Name', 'Ticket', 'Cabin'], axis=1)
test_data = test_data.drop(['Name', 'Ticket', 'Cabin'], axis=1)


if 'Survived' in test_data.columns:
    test_data = test_data.drop('Survived', axis=1)


train_data.head()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


X = train_data.drop('Survived', axis=1)
y = train_data['Survived']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_val)


accuracy = accuracy_score(y_val, y_pred)
conf_matrix = confusion_matrix(y_val, y_pred)
class_report = classification_report(y_val, y_pred)
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)


train_data = pd.read_csv('/kaggle/input/titanic-survivor-classification/train.csv')


survival_counts = train_data['Survived'].value_counts()


num_survived = survival_counts[1]
num_died = survival_counts[0]

print(f"Number of people who survived: {num_survived}")
print(f"Number of people who died: {num_died}")


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', data=train_data, palette='viridis')
plt.title('Number of People Who Survived and Died on the Titanic')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.xticks([0, 1], ['Died', 'Survived'])
plt.show()

import matplotlib.pyplot as plt

labels = ['Survived', 'Died']
sizes = [num_survived, num_died]
colors = ['#66b3ff', '#ff6666']
explode = (0.1, 0)  


plt.figure(figsize=(8, 8))
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Proportion of Survival on the Titanic')
plt.show()


train_data['AgeGroup'] = pd.cut(train_data['Age'], bins=[0, 12, 18, 35, 60, 100], labels=['Child', 'Teenager', 'Young Adult', 'Adult', 'Senior'])

plt.figure(figsize=(8, 6))
sns.countplot(x='AgeGroup', hue='Survived', data=train_data, palette='viridis')
plt.title('Survival Rate by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.legend(['Died', 'Survived'])
plt.show()

