import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import json
import time

# Page configuration
st.set_page_config(
    page_title="Celebrity Image Classifier",
    page_icon="👨‍🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Minimized to prevent conflicts with Streamlit's native styling
st.markdown("""
    <style>
    .stApp {
        max-width: 100%;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #cce5ff;
        border: 1px solid #b8daff;
        color: #004085;
        margin: 1rem 0;
    }
    .container {
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Helper functions
@st.cache_resource
def load_model():
    """Load the pre-trained model."""
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(load_class_names()))
    model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache_data
def load_class_names():
    """Load celebrity names from JSON."""
    with open('class_names.json', 'r') as json_file:
        return json.load(json_file)

def transform_image(image):
    """Process image for model input."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict(image, model, class_names):
    """Make prediction with confidence score."""
    image_tensor = transform_image(image)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        return class_names[predicted.item()], confidence.item()

# Sidebar
with st.sidebar:
    st.title("✨ About the App")
    
    # App description
    st.markdown("""
    This AI-powered tool identifies celebrities in your photos using advanced machine learning.
    """)
    
    # Features section
    st.subheader("📋 Features")
    st.markdown("""
    - Real-time celebrity recognition
    - High accuracy predictions
    - Support for JPG, JPEG, PNG
    - Confidence score display
    """)
    
    # Instructions section
    st.subheader("🔍 How to Use")
    st.markdown("""
    1. Upload your celebrity photo
    2. Wait for the AI analysis
    3. View the results and confidence score
    """)
    
    st.divider()
    
    # Tips section
    st.subheader("💡 Tips for Best Results")
    st.markdown("""
    - Use clear, well-lit photos
    - Ensure face is clearly visible
    - Avoid group photos
    - Higher resolution images work better
    """)

# Main content
st.title("Celebrity Image Classifier 👨‍🎤")

# Description container
st.markdown("""
Transform your photos into celebrity insights! Upload an image and let our AI tell you 
which celebrity it resembles the most.
""")

# Create two columns for the main content
left_col, right_col = st.columns([3, 2], gap="large")

with left_col:
    # File upload section
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )

    if uploaded_file:
        try:
            # Display uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Process image
            with st.spinner("Analyzing image..."):
                # Progress indication
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Load model and predict
                model = load_model()
                class_names = load_class_names()
                predicted_class, confidence = predict(image, model, class_names)
                
                # Clear progress bar after completion
                progress_bar.empty()
                
                # Results section
                st.subheader("🎯 Results")
                
                # Create metrics columns
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric(
                        label="Identified Celebrity",
                        value=predicted_class,
                        delta="Match Found" if confidence > 0.7 else "Low Confidence"
                    )
                with metric_col2:
                    st.metric(
                        label="Confidence Score",
                        value=f"{confidence*100:.1f}%",
                        delta="High" if confidence > 0.7 else "Low"
                    )
                
                # Additional details
                with st.expander("📊 Detailed Analysis"):
                    st.markdown(f"""
                    - **Prediction**: {predicted_class}
                    - **Confidence**: {confidence*100:.1f}%
                    - **Image Size**: {image.size}
                    - **Format**: {image.format if image.format else 'Unknown'}
                    """)
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.info("Please try uploading a different image.")
    else:
        # Placeholder when no image is uploaded
        st.info("👆 Please upload an image to begin the analysis")

with right_col:
    # Recent Updates or Featured Content
    st.subheader("💫 Featured Information")
    
    # Information cards using expanders
    with st.expander("ℹ️ About the Model", expanded=True):
        st.markdown("""
        Our AI model is trained on thousands of celebrity images and can recognize 
        various public figures from different fields including:
        - Hollywood movie stars
        """)
    
    with st.expander("🎯 Accuracy Information"):
        st.markdown("""
        The confidence score indicates how sure the model is about its prediction:
        - 90-100%: Very High Confidence
        - 70-90%: High Confidence
        - 50-70%: Moderate Confidence
        - Below 50%: Low Confidence
        """)
    
    with st.expander("❓ Troubleshooting"):
        st.markdown("""
        If you're getting unexpected results:
        1. Ensure the image is clear and well-lit
        2. Try a different photo of the same celebrity
        3. Make sure the face is clearly visible
        4. Avoid images with multiple people
        """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center'>
    <p>Made with ❤️ by Aafi Malek | Last Updated: October 2024</p>
    <p>For questions or feedback, please contact: aafimalek2023@gmail.com</p>
</div>
""", unsafe_allow_html=True)