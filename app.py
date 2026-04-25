import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# dataset load
data = pd.read_csv("insurance_claims.csv")

# simple preprocessing
data = data.dropna()

# example target column (fraud or not)
# make sure your csv la "fraud" column irukanum (0 or 1)
X = data.drop("fraud", axis=1)
y = data["fraud"]

# convert categorical to numeric
X = pd.get_dummies(X)

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# prediction sample
pred = model.predict(X_test)

print("Prediction:", pred[:5])
import joblib
joblib.dump(model, "fraud_model.pkl")
if __name__ == "__main__":
    app.run(debug=True)