# Import required libraries
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import joblib

# Load the trained model
dtc = joblib.load("decision_tree_model.pkl")

# Plot the decision tree
plt.figure(figsize=(20,10))
plot_tree(dtc, filled=True, feature_names=['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays', 'previous', 'poutcome'], class_names=['No', 'Yes'], rounded=True)
plt.show()