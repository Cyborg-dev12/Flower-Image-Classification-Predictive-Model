

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model selection - Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt='g', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Visualization Feature Importance (Optional)
feature_importances = model.feature_importances_
sns.barplot(x=iris.feature_names, y=feature_importances)
plt.title("Feature Importance")
plt.show()


import streamlit as st

# Title and description
st.title("Iris Dataset Prediction")
st.write("""
         This is a simple web app to classify Iris flower species based on the input features.
         """)

# Input sliders for features
sepal_length = st.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0, step=0.1)
sepal_width = st.slider("Sepal Width (cm)", min_value=2.0, max_value=5.0, step=0.1)
petal_length = st.slider("Petal Length (cm)", min_value=1.0, max_value=7.0, step=0.1)
petal_width = st.slider("Petal Width (cm)", min_value=0.1, max_value=3.0, step=0.1)

# Button for prediction
if st.button("Predict"):
    # Normalize the input using the scaler
    user_input = scaler.transform([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Prediction
    prediction = model.predict(user_input)
    prediction_species = iris.target_names[prediction][0]
    
    st.write(f"The predicted species is **{prediction_species}**.")

# Show model accuracy
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")



