# Task 5: Decision Trees and Random Forests

## 📌 Objective

This task focuses on learning and applying **tree-based models** — specifically, **Decision Trees** and **Random Forests** — for classification using the **Heart Disease Dataset**.

---

## 🧰 Tools & Libraries

- Python
- pandas, numpy
- scikit-learn (for model building)
- matplotlib, seaborn (for visualization)
- Graphviz (optional, for advanced tree visualization)

---

## 📁 Dataset

- **File**: `heart.csv`
- **Columns**:

| Feature      | Description                          |
|--------------|--------------------------------------|
| age          | Age of the patient                   |
| sex          | Sex (1 = male; 0 = female)           |
| cp           | Chest pain type (0–3)                |
| trestbps     | Resting blood pressure               |
| chol         | Serum cholesterol in mg/dl           |
| fbs          | Fasting blood sugar > 120 mg/dl      |
| restecg      | Resting electrocardiographic results |
| thalach      | Maximum heart rate achieved          |
| exang        | Exercise induced angina              |
| oldpeak      | ST depression induced by exercise    |
| slope        | Slope of the peak exercise ST segment|
| ca           | Number of major vessels (0–3)        |
| thal         | Thalassemia                          |
| target       | Target variable (1 = disease, 0 = no) |

---

## 📝 Tasks Performed

### 1. ✅ Train a Decision Tree Classifier
- Fit a basic decision tree.
- Visualize the tree using `plot_tree()`.

### 2. 📉 Analyze Overfitting & Control Depth
- Evaluate model performance across tree depths (1 to 20).
- Plot training vs testing accuracy.

### 3. 🌳 Train a Random Forest
- Fit a Random Forest model with 100 trees.
- Compare accuracy with Decision Tree.

### 4. 📊 Interpret Feature Importances
- Plot feature importances from Random Forest.
- Identify key predictors of heart disease.

### 5. 🔁 Cross-Validation Evaluation
- Use 5-fold cross-validation.
- Report mean and standard deviation of accuracy.

---

## 📈 Recommended Visualizations

| Plot | Purpose |
|------|---------|
| 🎄 Decision Tree Plot | Understand decision rules and splits |
| 📈 Accuracy vs Depth  | Analyze overfitting/underfitting tradeoff |
| 📊 Feature Importances | Identify key factors influencing prediction |

---

## 🚀 How to Run

```bash
# Step 1: Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Step 2: Run the script
heart.py
