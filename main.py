from groq import Groq
import streamlit as st
from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image, ImageDraw
import torch

llm = Groq(api_key="APIKEY")  # Replace with your actual Groq API key

# Load the YOLOs model and image processor
model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

# Streamlit app title
st.title("YOLOs Object Detection with Groq Summarization")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Process the image and run object detection
    st.write("Detecting objects...")
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Process the results
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

    # Extract detection results
    detection_results = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        detection_results.append(
            f"Detected {model.config.id2label[label.item()]} with confidence {round(score.item(), 3)} at location {box.tolist()}"
        )

    # Send detection results to Groq for summarization
    groq_prompt = "Summarize the following object detection results: " + "; ".join(detection_results)
    response = llm.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You are an AI specialized in summarizing object detection results."},
            {"role": "user", "content": groq_prompt}
        ]
    )

    # Extract the content of the response
    response_content = response.choices[0].message.content

    # Display the summarized results
    st.write("Summary of Detection Results:")
    st.write(response_content)

    # Draw bounding boxes on the image
    draw = ImageDraw.Draw(image)
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1]), f"{model.config.id2label[label.item()]}: {round(score.item(), 3)}", fill="red")

    # Display the image with bounding boxes
    st.image(image, caption='Detected Objects', use_column_width=True)
