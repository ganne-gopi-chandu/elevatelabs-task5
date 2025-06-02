import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import graphviz

# Load dataset
df = pd.read_csv('heart.csv')

# Define features and target
X = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
        'exang', 'oldpeak', 'slope', 'ca', 'thal']]  # Feature columns
y = df['target']  # Target column

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

# Visualize Decision Tree
dot_data = export_graphviz(dt_model, out_file=None, feature_names=X.columns, class_names=['No Disease', 'Disease'], filled=True)
graph = graphviz.Source(dot_data)
graph.render("decision_tree")  # Saves as a file

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions & Evaluation
y_pred_dt = dt_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

# Convert accuracy to percentage format
dt_accuracy = accuracy_score(y_test, y_pred_dt) * 100
rf_accuracy = accuracy_score(y_test, y_pred_rf) * 100

print(f"Decision Tree Accuracy: {dt_accuracy:.2f}%")
print(f"Random Forest Accuracy: {rf_accuracy:.2f}%")

print("\nDecision Tree Classification Report:\n", classification_report(y_test, y_pred_dt))
print("\nRandom Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

# Cross-validation in percentage
cv_dt = np.mean(cross_val_score(dt_model, X, y, cv=5)) * 100
cv_rf = np.mean(cross_val_score(rf_model, X, y, cv=5)) * 100
print(f"Decision Tree CV Score: {cv_dt:.2f}%")
print(f"Random Forest CV Score: {cv_rf:.2f}%")

# Feature Importance
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf_model.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances, palette='viridis')
plt.title("Feature Importances in Random Forest")
plt.show()
# Store accuracies
train_accuracies = []
test_accuracies = []

depths = range(1, 21)  # Vary max_depth from 1 to 20

for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)

    # Calculate training and testing accuracy
    train_acc = accuracy_score(y_train, model.predict(X_train)) * 100
    test_acc = accuracy_score(y_test, model.predict(X_test)) * 100

    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

# Plot training vs testing accuracy
plt.figure(figsize=(10, 6))
plt.plot(depths, train_accuracies, marker='o', label='Training Accuracy', color='blue')
plt.plot(depths, test_accuracies, marker='s', label='Testing Accuracy', color='red')
plt.xlabel('Tree Depth')
plt.ylabel('Accuracy (%)')
plt.title('Overfitting Analysis: Decision Tree Depth vs Accuracy')
plt.legend()
plt.show()
