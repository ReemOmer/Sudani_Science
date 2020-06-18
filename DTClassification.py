# Importing required libraries
from sklearn import tree

# Input
# 0 >> bumby, 1 >> smooth
features = [[150, 0], [170, 0], [140, 1], [130, 1]]

# Output
# labels = ['Orange', 'Orange',  'Apple', 'Apple'] was encoded to 0 >> orange, 1 >> apple 
labels = [0, 0, 1, 1]

# ML model, empty rule box
clf = tree.DecisionTreeClassifier()

# Training algorithm (to find patterns in data)
clf = clf.fit(features, labels)

# Make prediction based on the learned rules
print (clf.predict([[145,1]]))
