import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from model import TrafficSignNet

# ----- Label mapping for GTSRB -----
label_map = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons"
}

# ----- Load model -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TrafficSignNet().to(device)
model.load_state_dict(torch.load("traffic_sign_model.pth", map_location=device))
model.eval()

# ----- Define transforms -----
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ----- Streamlit UI -----
st.title("🚦 Traffic Sign Classifier")
st.markdown("Upload an image of a traffic sign (.jpg, .png, .ppm) to predict its class.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "ppm"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            pred_class = outputs.argmax(1).item()
            label = label_map.get(pred_class, "Unknown")

        st.success(f"**Predicted Class:** {pred_class} — _{label}_")

    except Exception as e:
        st.error(f"Error: {str(e)}")
