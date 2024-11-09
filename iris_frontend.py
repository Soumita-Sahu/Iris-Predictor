import streamlit as st
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import base64
from PIL import Image
import io

# Set page config for a more polished layout
st.set_page_config(page_title="🌸 Iris Species Prediction", layout="wide")

# Crop and resize image
def crop_image(image_path, crop_box):
    image = Image.open(image_path)
    cropped_image = image.crop(crop_box)
    return cropped_image

# Load and crop image for header
crop_box = (100, 250, 600, 400)
cropped_image = crop_image("iris.jpg", crop_box)

# Convert cropped image to base64
buffered = io.BytesIO()
cropped_image.save(buffered, format="JPEG")
img_base64 = base64.b64encode(buffered.getvalue()).decode()

# Display header with cropped image
st.markdown(
    f"""
    <style>
    .header-image {{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-size: cover;
        height: 200px;
        width: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    .header-text {{
        color: white;
        font-size: 24px;
        font-weight: bold;
        background-color: rgba(0, 0, 0, 0.6);
        padding: 10px 20px;
        border-radius: 10px;
    }}
    </style>
    <div class="header-image">
        <div class="header-text">🌸 Let's Predict the Iris Species! 🌼</div>
    </div>
    """,
    unsafe_allow_html=True
)

# Title and description
st.title('🌸 Iris Species Prediction 🌸')
st.markdown('This application predicts the species of Iris flowers based on sepal and petal measurements. 🌱')

# "Learn More" expandable section
with st.expander("Learn More about the Iris Model and Species 🌱"):
    st.write("### The Iris Dataset 📊:")
    st.markdown("""
    - The Iris dataset is one of the most famous datasets in machine learning and statistics, containing 150 samples of iris flowers with three species: **Setosa**, **Versicolor**, and **Virginica**.
    - It has four key features: **Sepal Length**, **Sepal Width**, **Petal Length**, and **Petal Width** which help differentiate between the species.
    - This dataset is frequently used as a beginner’s example for machine learning, helping to demonstrate classification algorithms.
    """)

    st.write("### Model Information 🔍:")
    st.markdown("""
    - The prediction model in this app is based on a **Random Forest Classifier** trained on the Iris dataset to classify the species based on the input measurements.
    - The model was evaluated and tuned for accuracy to ensure reliable predictions for most input ranges.
    """)

    st.write("### Understanding Prediction Confidence 🔮:")
    st.markdown("""
    - The **confidence level** represents how certain the model is about its prediction, shown as a percentage.
    - For instance, a confidence of 95% for a predicted species means that the model is quite certain of its prediction.
    """)

# Sidebar sliders for input parameters
st.sidebar.header('Input Parameters 🧑‍🔬')
sepal_length = st.sidebar.slider('Sepal Length (cm) 🌿', 4.0, 8.0, 5.0, key='sepal_length')
sepal_width = st.sidebar.slider('Sepal Width (cm) 🌿', 2.0, 5.0, 3.0, key='sepal_width')
petal_length = st.sidebar.slider('Petal Length (cm) 🌼', 1.0, 6.9, 3.0, key='petal_length')
petal_width = st.sidebar.slider('Petal Width (cm) 🌼', 0.0, 2.5, 1.0, key='petal_width')

# Input data for model
input_data = {
    "features": [float(sepal_length), float(sepal_width), float(petal_length), float(petal_width)]
}

# Display input data
st.write("### Input Data 📝:")
st.write(input_data)

# Make a request to model and display prediction with styling
st.write("### Model Prediction 🔮:")
try:
    response = requests.post('http://localhost:8000/predict/', json=input_data)
    response.raise_for_status()
    prediction = response.json().get("class", "Unknown")
    prediction_confidence = response.json().get("confidence", 0.95)

    # Styled card with a darker, semi-transparent background
    st.markdown(
        f"""
        <style>
            .prediction-card {{
                background-color: rgba(255, 255, 255, 0.1);
                padding: 20px;
                border-radius: 10px;
                border: 2px solid #4CAF50;
                text-align: center;
                color: #4CAF50;
                font-size: 20px;
            }}
            .prediction-header {{
                font-weight: bold;
                color: #FFD700;
                font-size: 24px;
            }}
            .confidence {{
                font-size: 18px;
                color: #00FA9A;
            }}
        </style>
        <div class="prediction-card">
            <div class="prediction-header">Predicted Species: {prediction} 🌺</div>
            <p class="confidence">Confidence Level: {prediction_confidence * 100:.2f}%</p>
        </div>
        """,
        unsafe_allow_html=True
    )

except requests.exceptions.RequestException as e:
    st.error(f"⚠️ Error occurred: {e}")
    st.write("Please ensure that the backend model is running and accessible.")

# Display visualizations in columns
col1, col2, col3 = st.columns(3)

# Sepal Characteristics Scatter Plot
with col1:
    st.write("### Sepal Characteristics 📊")
    iris_df = sns.load_dataset("iris")
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', style='species', s=70, palette='coolwarm', data=iris_df, ax=ax)
    ax.set_title('Sepal Length vs Sepal Width by Species', fontsize=14, color="darkblue")
    ax.set_xlabel("Sepal Length (cm)", fontsize=10)
    ax.set_ylabel("Sepal Width (cm)", fontsize=10)
    st.pyplot(fig)

# Species Prediction Distribution Pie Chart
with col2:
    st.write("### Species Prediction Distribution 📊")
    species_labels = ['Setosa', 'Versicolor', 'Virginica']
    species_data = [10, 30, 60]  # Example data, replace with actual probabilities
    colors = sns.color_palette("pastel")[0:3]
    fig, ax = plt.subplots(figsize=(4, 4))
    wedges, texts, autotexts = ax.pie(
        species_data, labels=species_labels, autopct='%1.1f%%', startangle=90, colors=colors,
        textprops={'color': "black"}
    )
    for text in texts + autotexts:
        text.set_fontsize(10)
    plt.setp(autotexts, size=10, weight="bold", color="white")
    ax.set_title('Species Distribution', fontsize=14, color="darkblue")
    ax.axis('equal')
    st.pyplot(fig)

# Petal Characteristics Histogram
with col3:
    st.write("### Petal Characteristics 📊")
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.histplot(iris_df, x="petal_length", hue="species", multiple="stack", palette="bright", kde=True)
    ax.set_title("Distribution of Petal Length by Species", fontsize=14, color="darkblue")
    st.pyplot(fig)

# User Feedback Section (Ratings and Comments)
st.write("### Your Feedback 💬:")
feedback_rating = st.slider("How was your experience with this site?", 1, 5, 4)
st.write(f"Thank you for your rating: {feedback_rating} ⭐")

# Add a text input for user comments
user_comments = st.text_area("Any suggestions or comments? 🤔")
if user_comments:
    st.write("Thank you for your feedback! We'll review your comments.")

# Display helpful tips
st.write("### Helpful Tips 📝:")
st.markdown("""
 - 🌱 The Iris dataset contains 3 species: **Setosa**, **Versicolor**, and **Virginica**.
 - 🌺 Adjust the sliders to test different values for Iris prediction.
 - 📊 View the scatter plot and pie chart for better insights.
 - 💬 Provide feedback on your experience.
""")
