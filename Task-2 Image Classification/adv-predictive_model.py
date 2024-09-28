
# import numpy as np
# import pandas as pd
# import streamlit as st
# from sklearn.datasets import load_iris
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler

# @st.cache_data
# def load_and_preprocess_data():
#     iris = load_iris()
#     X = pd.DataFrame(iris.data, columns=iris.feature_names)
#     y = pd.DataFrame(iris.target, columns=["species"])
    
   
#     X.iloc[::15, 0] = np.nan
#     X.fillna(X.mean(), inplace=True) 
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#     return X_scaled, y.values.ravel(), iris.target_names, iris.feature_names
# @st.cache_data
# def train_model(X, y):
#     model = RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=5, random_state=42)
#     model.fit(X, y)
#     return model
# def main():
#     st.title("Iris Flower Classification 🌸")
#     st.write("""
#         This app uses a machine learning model (Random Forest) to predict the species of an iris flower
#         based on user input features such as **sepal length, sepal width, petal length, and petal width**.
#     """)

#     X, y, target_names, feature_names = load_and_preprocess_data()
#     model = train_model(X, y)

#     st.header("Input flower measurements")
#     sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
#     sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
#     petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
#     petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

    
#     user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
#     scaled_input = StandardScaler().fit_transform(user_input)  # Scale user input
    
    
#     if st.button("Predict"):
#         prediction = model.predict(scaled_input)
#         predicted_species = target_names[prediction][0]
#         st.success(f"Predicted species: **{predicted_species}**")

    
#     st.sidebar.header("About the Model")
#     st.sidebar.write(f"Accuracy on Training Data: {model.score(X, y) * 100:.2f}%")
#     st.sidebar.write(f"Model: Random Forest with 50 estimators")

# if __name__ == "__main__":
#     main()


import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array

@st.cache_resource
def load_model():
    model = MobileNetV2(weights="imagenet")
    return model

def preprocess_image(image, target_size=(224, 224)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)  
    return image_array

def main():
    st.set_page_config(page_title="Flower Image Classifier 🌸", layout="centered", initial_sidebar_state="expanded")

    st.title("🌸 Flower Image Classification App")
    st.write("Upload a flower image and the app will predict its species using a pre-trained MobileNetV2 model.")

    model = load_model()
    uploaded_file = st.file_uploader("Choose a flower image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Processing image...")
        preprocessed_image = preprocess_image(image)
        st.write("Classifying image...")
        predictions = model.predict(preprocessed_image)
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        st.subheader("🔮 Prediction Results")
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            st.write(f"{i+1}. **{label.capitalize()}** with a confidence of **{score * 100:.2f}%**.")

if __name__ == "__main__":
    main()
