import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
from fpdf import FPDF
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import base64
import requests
import tempfile
import os

# Load model
model = load_model('terrain__2023_09_13__11_52_06___Accuracy_0.9787.h5')

# Class labels
label_map = {0: 'Sandy', 1: 'Marshy', 2: 'Rocky', 3: 'Grassy'}

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

class PDF(FPDF):
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, body, image_path):
        # Add an image
        self.image(image_path, x=10, w=80)

        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

def classify_image(img):
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    prediction = model.predict(x)
    label_index = np.argmax(prediction)
    prediction_prob = prediction[0, label_index]
    return label_map[label_index], prediction_prob

def classify_image_from_url(image_url):
    try:
        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            img = Image.open(response.raw)
            img = img.resize((224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            prediction = model.predict(x)
            label_index = np.argmax(prediction)
            prediction_prob = prediction[0, label_index]
            return label_map[label_index], prediction_prob
        else:
            return "Unknown", 0.0
    except Exception as e:
        return "Error", 0.0

def generate_pdf_report(df):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "EarthFinesse Military Terrain Classification Report", ln=True, align="C")
    pdf.ln(10)
    pdf.rect(5.0, 5.0, 200.0, 280.0)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, "Bulk Classification Results", ln=True, align="L")
    pdf.ln(10)

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pdf_file_name = f"EarthFinesse_Classification_Report_{current_datetime}.pdf"

    for i, (_, row) in enumerate(df.iterrows()):
        terrain, confidence = row["Terrain"], row["Confidence"]
        image_path = row["Image"]
        body = f"Terrain: {terrain}\nConfidence: {confidence:.2%}"

        pdf.chapter_body(body, image_path)

    pdf_file_path = os.path.join("pdf_reports", pdf_file_name)
    os.makedirs("pdf_reports", exist_ok=True)

    pdf.output(pdf_file_path)

    return pdf_file_path


def main():
    st.sidebar.title('Select Operation')
    operation = st.sidebar.radio("Choose an operation:", ("File Upload", "Map Coordinates"))

    if operation == "File Upload":
        st.title('🌍 EarthFinesse - Military Terrain Classifier 🛡️')
        st.header('File Upload and Classification')
        
        st.sidebar.header('Mission Settings')
        threshold = st.sidebar.slider('Confidence Threshold', 0.0, 1.0, 0.5, 0.01)
        show_probabilities = st.sidebar.checkbox('Show Probabilities', False)
        bulk_classification = st.sidebar.checkbox('Bulk Classification', False)

        if bulk_classification:
            st.sidebar.write("Upload multiple images to classify in bulk.")
        else:
            st.sidebar.write("Upload a single image for classification.")

        uploaded_files = st.file_uploader('Upload Reconnaissance Images', type=['png', 'jpg'], accept_multiple_files=True, help="Select one or more reconnaissance images.")

        if uploaded_files:
            bulk_results = []
            st.markdown("---")

            for i, file in enumerate(uploaded_files):
                st.subheader(f"Image {i+1}")
                st.image(file, caption="Reconnaissance Image", use_column_width=True)

                terrain, confidence = classify_image(Image.open(file))

                # Save the image to a temporary file in PNG format
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_image:
                    img = Image.open(file)
                    img.save(temp_image, format="PNG")
                    temp_image_path = temp_image.name

                bulk_results.append({
                    'Image': temp_image_path,
                    'Terrain': terrain,
                    'Confidence': confidence
                })

            if bulk_results:
                st.sidebar.markdown("---")
                st.sidebar.header('Generate PDF Report')
                if st.sidebar.button("Download PDF Report"):
                    pdf_file_path = generate_pdf_report(pd.DataFrame(bulk_results))
                    st.sidebar.success(f"PDF report generated! [Download PDF]({pdf_file_path})")

    else:
        st.title('🌍 EarthFinesse - Military Terrain Classifier 🛡️')
        st.header('Map Coordinates and Terrain Classification')
        
        st.sidebar.header('Mission Settings - Map Coordinates')
        place_name = st.sidebar.text_input("Enter a location (e.g., city or landmark):")
        zoom_level = st.sidebar.slider("Zoom Level", 1, 18, 10)
        search_button = st.sidebar.button("Search")

        if search_button and place_name:
            try:
                # Use OpenStreetMap Nominatim API to get the coordinates of the place
                osm_api_url = f"https://nominatim.openstreetmap.org/search?format=json&q={place_name}"
                response = requests.get(osm_api_url)
                data = response.json()

                if len(data) > 0:
                    latitude = float(data[0]['lat'])
                    longitude = float(data[0]['lon'])

                    st.subheader(f"Location Details:")
                    st.write(f"Latitude: {latitude}")
                    st.write(f"Longitude: {longitude}")

                    st.subheader(f"🛰️ Satellite Imagery of {place_name}:")
                    google_maps_url = f"https://www.google.com/maps/embed?pb=!1m14!1m12!1m3!1d28961.509524683654!2d{longitude}!3d{latitude}!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!5e1!3m2!1sen!2sus!4v1630323443061!5m2!1sen!2sus"
                    st.markdown(f'<iframe width="600" height="450" frameborder="0" style="border:0" src="{google_maps_url}" allowfullscreen></iframe>', unsafe_allow_html=True)

                    terrain, confidence = classify_image_from_url(google_maps_url)
                    st.subheader('🌲 Predicted Terrain Type:')
                    st.write(f'🌿 {terrain}')
                    st.subheader('🎯 Prediction Accuracy:')
                    st.write(f'📊 {confidence * 100:.2f}%')

                else:
                    st.error(f"No coordinates found for the provided location: {place_name}")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
