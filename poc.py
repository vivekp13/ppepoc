import streamlit as st
import torch
import clip
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F

# Load CLIP model for PPE detection [[9]]
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Define PPE prompts [[2]][[3]]
ppe_prompts = [
    "a person wearing a hard hat",
    "a person wearing safety goggles",
    "a person wearing flame-resistant gloves",
    "a person wearing high-visibility vest"
]
text_inputs = torch.cat([clip.tokenize(prompt) for prompt in ppe_prompts]).to(device)

# Load Faster R-CNN for person detection [[8]]
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
person_detector = fasterrcnn_resnet50_fpn(weights=weights).to(device)
person_detector.eval()

# Transform for Faster R-CNN
transform = weights.transforms()

# Streamlit UI
st.title("Refinery PPE Detection & Compliance Dashboard [[7]]")

# Sidebar for file upload
st.sidebar.header("Upload Files")
uploaded_file = st.sidebar.file_uploader("Upload Image or Video", type=["jpg", "png", "mp4"])

# Initialize compliance data
compliance_data = []

# Function to detect persons in a frame
def detect_persons(frame):
    img = F.to_tensor(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        predictions = person_detector(img)[0]
    return predictions

# Function to process a single person's PPE
def process_person(person_crop):
    image = Image.fromarray(person_crop)
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_inputs)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    scores = similarity[0].cpu().numpy()
    return scores.max() >= 0.7  # Check compliance

# Process uploaded file
if uploaded_file:
    file_type = uploaded_file.type.split("/")[0]
    
    if file_type == "image":
        # Process image
        image = Image.open(uploaded_file).convert("RGB")
        frame = np.array(image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Detect persons
        predictions = detect_persons(frame)
        boxes = predictions["boxes"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()
        
        compliant_count = 0
        total_persons = 0
        
        for box, label, score in zip(boxes, labels, scores):
            if label == 1 and score > 0.5:  # Label 1 corresponds to 'person'
                total_persons += 1
                x1, y1, x2, y2 = map(int, box)
                person_crop = frame[y1:y2, x1:x2]
                
                # Check compliance
                compliant = process_person(person_crop)
                compliant_count += int(compliant)
                
                # Draw bounding box
                color = (0, 255, 0) if compliant else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Display results
        st.subheader("PPE Detection Results:")
        st.write(f"Total Persons Detected: {total_persons}")
        st.write(f"Compliant Persons: {compliant_count}")
        st.write(f"Non-Compliant Persons: {total_persons - compliant_count}")
        st.image(frame, caption="Processed Image", channels="BGR")
        
        # Add to compliance data
        compliance_data.append({
            "File": uploaded_file.name,
            "Total Persons": total_persons,
            "Compliant Persons": compliant_count,
            "Non-Compliant Persons": total_persons - compliant_count
        })
    
    elif file_type == "video":
        # Process video
        st.subheader("Processing Video...")
        video_bytes = uploaded_file.read()
        temp_file = f"temp_video.{uploaded_file.name.split('.')[-1]}"
        with open(temp_file, "wb") as f:
            f.write(video_bytes)
        
        cap = cv2.VideoCapture(temp_file)
        frame_count = 0
        compliant_frames = 0
        total_persons = 0
        non_compliant_persons = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1
            
            # Detect persons
            predictions = detect_persons(frame)
            boxes = predictions["boxes"].cpu().numpy()
            labels = predictions["labels"].cpu().numpy()
            scores = predictions["scores"].cpu().numpy()
            
            frame_total = 0
            frame_compliant = 0
            
            for box, label, score in zip(boxes, labels, scores):
                if label == 1 and score > 0.5:  # Label 1 corresponds to 'person'
                    frame_total += 1
                    total_persons += 1
                    x1, y1, x2, y2 = map(int, box)
                    person_crop = frame[y1:y2, x1:x2]
                    
                    # Check compliance
                    compliant = process_person(person_crop)
                    frame_compliant += int(compliant)
                    compliant_frames += int(compliant)
                    
                    # Draw bounding box
                    color = (0, 255, 0) if compliant else (255, 0, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            non_compliant_persons += frame_total - frame_compliant
            
            # Display progress
            if frame_count % 30 == 0:  # Update every 30 frames
                st.progress(min(1.0, frame_count / int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
        
        cap.release()
        compliance_rate = compliant_frames / total_persons if total_persons > 0 else 0
        compliance_data.append({
            "File": uploaded_file.name,
            "Total Persons": total_persons,
            "Compliant Persons": compliant_frames,
            "Non-Compliant Persons": non_compliant_persons,
            "Compliance Rate": compliance_rate
        })
        st.success(f"Video processed! Compliance Rate: {compliance_rate*100:.1f}%", icon="✅")
    
    # Remove temporary file
    if file_type == "video":
        import os
        os.remove(temp_file)

# Compliance Dashboard
st.sidebar.header("Compliance Dashboard")
if compliance_data:
    df = pd.DataFrame(compliance_data)
    st.sidebar.dataframe(df)
    st.sidebar.bar_chart(df.set_index("File")["Compliance Rate"] if "Compliance Rate" in df.columns else None)
else:
    st.sidebar.write("No compliance data available yet.")