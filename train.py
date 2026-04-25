import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. dataset load
data = pd.read_csv("insurance_claims.csv")

# 2. basic cleaning
data = data.dropna()

# 3. features & target
X = data.drop("fraud", axis=1)
y = data["fraud"]

# 4. convert text to number
X = pd.get_dummies(X)

# 5. split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 6. model train
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 7. save model
joblib.dump(model, "fraud_model.pkl")

print("Model trained and saved successfully 👍")