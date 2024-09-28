Flower Classification Model 🌸

Overview:

This repository contains a machine learning-based Flower Classification Model designed to classify flowers into species using two approaches:

  Random Forest Classifier: For numerical input features like sepal/petal length and width (based on the Iris dataset).
  
  MobileNetV2 CNN: For image-based flower classification using a pre-trained deep learning model.

Features:

Structured Data Classification: Input features include sepal and petal dimensions.

Image Classification: Upload flower images to predict species.

High Accuracy: Both models achieve high accuracy, making the app useful for botanical studies, gardening tools, or educational purposes.

How to Use:

Clone the repository:
	
	git clone https://github.com/your-username/flower-classification.git
Set up the environment and install dependencies:

	pip install -r requirements.txt
Run the app (Streamlit-based for easy interaction):

	streamlit run app.py
Requirements

	Python 3.9+
	TensorFlow
	Scikit-learn
	Streamlit

Model Details

    Random Forest Classifier for tabular data (Iris dataset).
    MobileNetV2 for image-based classification.
