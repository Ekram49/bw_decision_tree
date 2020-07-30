from sklearn.tree import DecisionTreeClassifier
import pandas as pd

train = pd.read_csv("./data/train_preprocessed.csv")
test = pd.read_csv("./data/test_preprocessed.csv")


#train_features = 
model = DecisionTreeClassifier()
model.fit(train.drop(columns = "Survived"), train["Survived"])
predictions = model.predict(test)
print(predictions[:10])