# ml4

import numpy as n
import pandas as p
import matplotlib.pyplot as plt

data=p.read_csv(r'C:\Users\hmr\Desktop\wine_dataset.csv')
print(data)
y=data['Proline']

from sklearn.model_selection import train_test_split
X_train, X_test,Y_train,Y_test= train_test_split(data.values,data.Proline)
from sklearn.linear_model import LogisticRegression
clf= LogisticRegression()

clf.fit(X_train,Y_train)
predicted=clf.predict(X_test)
expected= Y_test

plt.figure(figsize=(4,3))
plt.scatter(expected,predicted)
#plt.plot([10,50],[0,50],'--k')
plt.axis('tight')
#plt.xlabel('True price')
#plt.ylabel('predicted price')
plt.tight_layout()
plt.show()

"""
a=data.values
x=a[:,0:12]
print(x)
plt.hist(x)
plt.show()
print(data.columns)
print(data['Alcohol'])
plt.hist(data['Alcohol'])
print(data['Malic acid'])
plt.hist(data['Malic acid'])
print(data['Ash'])
plt.hist(data['Ash'])
print(data['Alcalinity of ash'])
plt.hist(data['Alcalinity of ash'])
print(data['Magnesium'])
plt.hist(data['Magnesium'])
print(data['Total phenols'])
plt.hist(data['Total phenols'])
print(data['Flavanoids'])
plt.hist(data['Flavanoids'])
print(data['Nonflavanoid phenols'])
plt.hist(data['Nonflavanoid phenols'])
print(data['Proanthocyanins'])
plt.hist(data['Proanthocyanins'])
print(data['Color intensity'])
plt.hist(data['Color intensity'])
print(data['Hue'])
plt.hist(data['Hue'])
print(data['OD280/OD315 of diluted wines'])
plt.hist(data['OD280/OD315 of diluted wines'])

print(data['Proline'])
plt.hist(data['Proline'])

print(data['Wine Type'])
plt.hist(data['Wine Type'])

new_data=n.array(data)

print(new_data[0])
plt.show()
"""
 
