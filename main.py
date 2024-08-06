import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from PIL import Image

# Define model architecture to match the trained model
def load_model():
    model = models.efficientnet_b4(pretrained=False)
    num_classes = 5  # Change to the number of classes in your dataset
    
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.BatchNorm1d(model.classifier[1].in_features),
        nn.Linear(model.classifier[1].in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.Linear(128, num_classes)
    )

    model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Load the model
model = load_model()

# Define transformations
transforms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define class names
class_names = ['heart', 'oblong', 'oval', 'round', 'square']  # Replace with your class names

st.title("Face Shape Classification")
st.write("Upload an image to classify its face shape.")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess image
    image = transforms(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)

    st.write(f'Predicted class: {class_names[predicted.item()]}')
