import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import base64
from io import BytesIO
import os

# Streamlit page configuration
st.set_page_config(page_title="Potato Classifier", page_icon="🥔", layout="centered")

# Load and encode background image to base64
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()
        return f"data:image/jpeg;base64,{encoded}"
    except FileNotFoundError:
        st.error(f"Background image '{image_path}' not found in the root folder.")
        return None
    except Exception as e:
        st.error(f"Error loading background image: {str(e)}")
        return None

# Path to the background image in the root folder
background_image_path = "image2.jpg"
background_image_base64 = get_base64_image(background_image_path)

# Custom CSS for red-themed styling and background image
st.markdown(f"""
<style>
    .stApp {{
        background-color: #F9F9F9;
        font-family: 'Arial', sans-serif;
    }}
    .hero-section {{
        background-image: url('{background_image_base64 or "https://images.unsplash.com/photo-1518977822534-7049a61ee384?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80"}');
        background-size: cover;
        background-position: center;
        height: 50vh;
        width: 100%;
        display: flex;
        flex-direction: column;
        justify-content: flex-end; /* Align content to the bottom */
        align-items: center;
        position: relative;
        overflow: hidden;
        padding-bottom: 20px; /* Space from bottom edge */
    }}
    .hero-section::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0); /* 40% opacity black overlay */
        z-index: 1;
    }}
    .hero-section h1, .hero-section p {{
        position: relative;
        z-index: 2;
        color: #FFFFFF !important; /* White text for contrast */
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.5);
        margin: 0.5rem 0; /* Consistent spacing between title and description */
    }}
    .hero-section h1 {{
        font-size: 2.5rem !important;
    }}
    .hero-section p {{
        font-size: 1.1rem !important;
    }}
    h1 {{
        color: #D32F2F !important;
        text-align: center;
        font-size: 2.5rem !important;
    }}
    h2 {{
        color: #D32F2F !important;
        font-size: 1.5rem !important;
        margin-bottom: 0.5rem;
    }}
    .stMarkdown p {{
        color: #333333;
        font-size: 1.1rem;
    }}
    .stWarning {{
        background-color: #FFEBEE !important;
        border: 2px solid #D32F2F !important;
        color: #D32F2F !important;
        padding: 10px;
        border-radius: 8px;
        font-size: 1.1rem;
    }}
    .fresh-text {{
        color: #4CAF50 !important;
        font-weight: bold;
    }}
    .rotten-text {{
        color: #D32F2F !important;
        font-weight: bold;
    }}
    .sprouted-text {{
        color: #FFB300 !important;
        font-weight: bold;
    }}
    .green-text {{
        color: #26A69A !important;
        font-weight: bold;
    }}
    .bruised-text {{
        color: #8D6E63 !important;
        font-weight: bold;
    }}
    .stFileUploader label {{
        color: #D32F2F !important;
        font-size: 1.2rem !important;
    }}
    .confidence-text {{
        font-size: 1.2rem;
        margin: 0.5rem 0;
    }}
    .fresh-confidence {{
        color: #4CAF50 !important;
        font-weight: bold;
    }}
    .rotten-confidence {{
        color: #D32F2F !important;
        font-weight: bold;
    }}
    .sprouted-confidence {{
        color: #FFB300 !important;
        font-weight: bold;
    }}
    .green-confidence {{
        color: #26A69A !important;
        font-weight: bold;
    }}
    .bruised-confidence {{
        color: #8D6E63 !important;
        font-weight: bold;
    }}
    .stButton>button {{
        background-color: #D32F2F;
        color: #FFFFFF;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
        font-size: 1rem;
    }}
    .stButton>button:hover {{
        background-color: #B71C1C;
    }}
    .uploaded-image {{
        max-width: 300px !important;
        margin: auto;
        display: block;
    }}
</style>
""", unsafe_allow_html=True)

# Hero section with background image
st.markdown("""
<div class="hero-section">
    <h1>Potato Image Classifier 🥔</h1>
    <p>Upload an image of a potato to classify it as Fresh, Rotten, Sprouted, Green, or Bruised.</p>
</div>
""", unsafe_allow_html=True)

# Placeholder for loading single pre-trained model
@st.cache_resource
def load_model():
    # Initialize ResNet50 model for 5 classes
    model = tf.keras.applications.ResNet50(
        weights=None,  # Custom weights
        input_shape=(224, 224, 3),
        classes=5
    )
    # model.load_weights('path_to_resnet_weights.h5')
    return model

model = load_model()

# Class labels
class_labels = ['Fresh', 'Rotten', 'Sprouted', 'Green', 'Bruised']

# Confidence threshold for detecting irrelevant images
CONFIDENCE_THRESHOLD = 15.0

# Function to convert PIL image to base64 for HTML embedding
def pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# Image upload
uploaded_file = st.file_uploader("Choose a potato image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Create two columns for side-by-side display
    col1, col2 = st.columns([1.2, 1])

    # Display uploaded image in the first column
    with col1:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file)
        # Use markdown to display image with custom CSS
        img_base64 = pil_image_to_base64(image)
        st.markdown(f'<img src="{img_base64}" class="uploaded-image" alt="Uploaded Image">', unsafe_allow_html=True)

    # Preprocess image
    def preprocess_image(img):
        img = img.resize((224, 224))  # Resize to match model input
        img_array = np.array(img)
        if img_array.shape[-1] != 3:  # Convert to RGB if needed
            img_array = np.stack([img_array] * 3, axis=-1)
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array

    # Predict with single model
    with st.spinner("Classifying image..."):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        confidence = np.max(prediction[0]) * 100

    # Display results in the second column
    with col2:
        st.subheader("Classification Result")
        if confidence < CONFIDENCE_THRESHOLD:
            st.warning("This image may not be a potato! The model's confidence is too low.")
        else:
            for label, prob in zip(class_labels, prediction[0] * 100):
                class_style = ("fresh-confidence" if label == "Fresh" else
                              "rotten-confidence" if label == "Rotten" else
                              "sprouted-confidence" if label == "Sprouted" else
                              "green-confidence" if label == "Green" else
                              "bruised-confidence")
                st.markdown(f'<p class="confidence-text {class_style}">{label}: {prob:.2f}%</p>', unsafe_allow_html=True)

# Potato condition information
with st.expander("Potato Condition Guide"):
    st.markdown('<span class="fresh-text">**Fresh (Recommended)**</span>: Firm, clean, smooth skin; no sprouting, bruises, or green spots; ready for sale and consumption.', unsafe_allow_html=True)
    st.markdown('<span class="rotten-text">**Rotten**</span>: Soft, mushy, foul-smelling; black or brown discoloration; caused by bacterial or fungal decay.', unsafe_allow_html=True)
    st.markdown('<span class="sprouted-text">**Sprouted**</span>: Visible sprouts or green shoots; may be unsafe to eat; often salvageable if sprouts are removed.', unsafe_allow_html=True)
    st.markdown('<span class="green-text">**Green**</span>: Green skin or flesh due to sunlight exposure; may contain solanine, potentially toxic; trim green parts before use.', unsafe_allow_html=True)
    st.markdown('<span class="bruised-text">**Bruised**</span>: Dark spots or soft areas from physical damage; edible after removing affected parts; lower market value.', unsafe_allow_html=True)