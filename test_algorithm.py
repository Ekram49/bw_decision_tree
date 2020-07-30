
from decision_tree import DecisionTree
import pandas as pd

train = pd.read_csv("./data/train_preprocessed.csv")
test = pd.read_csv("./data/test_preprocessed.csv")

model = DecisionTree()
model.fit(data = train, target = "Survived")
predictions = model.predict(test)
print(predictions[:5])
