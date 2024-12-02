# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
data = pd.read_csv("bank.csv", sep=";")

# Display the first few rows of the dataset
print("Dataset Overview:")
print(data.head())

# Preprocess the data
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Define features (X) and target (y)
X = data.drop(columns=['y'])  # 'y' is the target column
y = data['y']  # Target: whether the customer subscribed to the product (binary: yes or no)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build the Decision Tree Classifier
dtc = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
dtc.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dtc.predict(X_test)

# Evaluate the model
print("Decision Tree Classifier Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(dtc, "decision_tree_model.pkl")
print("Decision Tree model saved as 'decision_tree_model.pkl'.")