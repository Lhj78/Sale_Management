import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("personality_dataset.csv")

# Step 1: Split into train and test
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)  # Adjust size as needed

# Step 2: Save train and test files
train_df.to_csv("train.csv", index=False)

# Drop target from test
test_no_labels = test_df.drop(columns=["personality_type"])
test_no_labels.to_csv("test.csv", index=False)

# Step 3: Train model on train data
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
X_train = train_df.drop(columns=["personality_type", "id"])
y_train = le.fit_transform(train_df["personality_type"])

X_test = test_no_labels.drop(columns=["id"])

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 4: Predict on test set
y_pred = model.predict(X_test)
predicted_labels = le.inverse_transform(y_pred)

submission = pd.DataFrame({
    "id": test_no_labels["id"],
    "personality_type": predicted_labels
})

# Step 5: Save submission file (MUST MATCH test.csv IDs and length)
submission.to_csv("submission.csv", index=False)
