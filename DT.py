import pydotplus
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets

iris = datasets.load_iris()
features = iris.data
target = iris.target

decisiontree = DecisionTreeClassifier(random_state=0)
model = decisiontree.fit(features, target)

observation = [[5, 4, 3, 2]]  # Predict observation's class
y = model.predict(observation)
model.predict_proba(observation)
print(y)
