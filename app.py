import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Page config
st.set_page_config(
    page_title='EarthFinesse - Military Terrain Classifier',
    page_icon='🌍',
    layout='wide',
    initial_sidebar_state='auto'
)

# Load model
model = load_model('terrain__2023_09_13__11_52_06___Accuracy_0.9787.h5')

# Class labels
label_map = {0: 'Grassy', 1: 'Marshy', 2: 'Rocky', 3: 'Sandy'}

# Define military camo theme colors
bg_color = "#383838"  # Dark background
text_color = "#FFFFFF"  # White text
accent_color = "#4CAF50"  # Military green accent color

# Apply the theme
st.markdown(
    f"""
    <style>
    .reportview-container {{
        background-color: {bg_color};
        color: {text_color};
    }}
    .sidebar .sidebar-content {{
        background-color: {bg_color};
    }}
    .widget-label {{
        color: {text_color};
    }}
    .stButton>button {{
        color: {text_color};
        background-color: {accent_color};
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Page title and intro
st.title('Welcome to EarthFinesse 🌍🛡️')
st.write('Military Terrain Classification Tool')

# Sidebar options
st.sidebar.header('Mission Settings')
threshold = st.sidebar.slider('Confidence Threshold', 0.0, 1.0, 0.5, 0.01, help="Adjust the prediction threshold for classifying images.")
show_probabilities = st.sidebar.checkbox('Show Probabilities', False, help="Display prediction probabilities along with terrain labels.")
bulk_classification = st.sidebar.checkbox('Bulk Classification', False, help="Enable batch classification for multiple images.")
if bulk_classification:
    st.sidebar.write("Upload multiple images to classify in bulk.")

# File uploader to accept multiple images
uploaded_files = st.file_uploader('Upload Recon Images', type=['png', 'jpg'], accept_multiple_files=True, help="Select one or more reconnaissance images.")

if uploaded_files:

    # Initialize an empty list to store bulk classification results
    bulk_results = []

    # Process each uploaded image
    for file in uploaded_files:

        # Load and preprocess image
        img = Image.open(file)
        img = img.resize((224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Make prediction
        prediction = model.predict(x)
        label_index = np.argmax(prediction)
        prediction_prob = prediction[0, label_index]

        # Determine if the prediction is uncertain
        uncertain = prediction_prob < 0.25

        # Display image and result
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(img, use_column_width=True, caption="Recon Image")
        with col2:
            if uncertain:
                st.error('Uncertain Terrain')
            elif prediction_prob >= threshold:
                st.success(f'Terrain: {label_map[label_index]}')
                if show_probabilities:
                    st.write(f'Confidence: {prediction_prob:.2%}')
            else:
                st.error('Uncertain Terrain')

        # Append the result to the bulk results list
        bulk_results.append({
            'FileName': file.name,
            'Terrain': label_map[label_index] if not uncertain else 'Uncertain Terrain',
            'Confidence': prediction_prob
        })

    # Offer bulk classification
    if bulk_classification:
        st.sidebar.header('Bulk Classification Report')
        df = pd.DataFrame(bulk_results)  # Create DataFrame from bulk results
        st.dataframe(df)

# Footer
st.sidebar.header('About EarthFinesse 🌍🛡️')
st.sidebar.write("EarthFinesse is a military terrain classification tool designed for reconnaissance purposes. It identifies terrain types, such as Grassy, Marshy, Rocky, and Sandy, in images.")
st.sidebar.write("Built for the defense community by \nThe Syntax Slingers")