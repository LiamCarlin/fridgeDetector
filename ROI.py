import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import urllib.request
import openai
import base64

# Set up your OpenAI API key
# It's recommended to store your API key in an environment variable named "OPENAI_API_KEY".
# For example, in your terminal or your IDE's run configuration:
# export OPENAI_API_KEY="your_actual_openai_api_key"
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    raise ValueError("No OpenAI API key provided. Please set the OPENAI_API_KEY environment variable.")

def load_imagenet_classes():
    """
    Loads the ImageNet class index mapping from a public URL.
    Returns:
        classes (list): A list where index corresponds to the class label.
    """
    url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    with urllib.request.urlopen(url) as f:
        class_idx = json.load(f)
    classes = [class_idx[str(k)][1] for k in range(len(class_idx))]
    return classes

def get_model():
    """
    Loads a pre-trained ResNet-18 model for image classification.
    Returns:
        model: The loaded model in evaluation mode.
    """
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    model.eval()
    return model

def classify_image(roi, model, classes, preprocess):
    """
    Classifies a region of interest (ROI) image.
    
    Args:
        roi (numpy.ndarray): The ROI image (in BGR format from OpenCV).
        model: Pre-trained image classification model.
        classes (list): List of class labels for ImageNet.
        preprocess: Preprocessing pipeline.
        
    Returns:
        predicted_label (str): The predicted class label.
    """
    # Convert ROI from BGR (OpenCV) to RGB (PIL)
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(roi_rgb)
    
    # Preprocess the image for the model
    input_tensor = preprocess(pil_image)
    input_batch = input_tensor.unsqueeze(0)  # add batch dimension

    # Get prediction from the model
    with torch.no_grad():
        output = model(input_batch)
    _, predicted_idx = torch.max(output, 1)
    predicted_label = classes[predicted_idx.item()]
    return predicted_label

def query_chatgpt(roi):
    # Convert ROI to base64
    _, buffer = cv2.imencode('.png', roi)
    roi_base64 = base64.b64encode(buffer).decode('utf-8')
    prompt = f"The image displayed is a picture inside of a fridge. Here is the image data in base64:\n{roi_base64}\nPlease use any references to figure out what the items in the picture are."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

def detect_objects_in_fridge(image_path):
    """
    Detects non-white objects in an image of a fridge by finding ROIs.
    
    Args:
        image_path (str): The file path to the image.
        
    Returns:
        rois (list): List of ROI images (numpy arrays).
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return []

    # Create a copy for drawing bounding boxes
    annotated_image = image.copy()

    # Define the range for white (background)
    lower_white = np.array([215, 215, 215], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)

    # Create a mask for white areas (the background)
    white_mask = cv2.inRange(image, lower_white, upper_white)

    # Invert the mask to isolate non-white regions (potential objects)
    objects_mask = cv2.bitwise_not(white_mask)

    # Apply morphological operations to reduce noise and fill holes
    kernel = np.ones((3, 3), np.uint8)
    objects_mask = cv2.morphologyEx(objects_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    objects_mask = cv2.morphologyEx(objects_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Debug: Display intermediate masks (optional)
    cv2.imshow("White Mask", white_mask)
    cv2.imshow("Objects Mask", objects_mask)

    # Find contours of the objects
    contours, _ = cv2.findContours(objects_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rois = []

    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        # Skip very small contours that might be noise; adjust the area threshold as needed
        if area < 100:
            continue

        # Get bounding rectangle for each contour and extract ROI
        x, y, w, h = cv2.boundingRect(contour)
        roi = image[y:y+h, x:x+w]
        rois.append(roi)

        # Draw bounding box and label on the annotated image
        cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(annotated_image, f'Obj {idx+1}', (x, y-10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the annotated image with detected objects
    cv2.imshow("Detected Objects", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return rois

if __name__ == '__main__':
    # Path to your fridge image
    image_path = 'fridge_items_2.webp'
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        exit()

    # Initialize the image classification model, preprocessing, and class labels
    model = get_model()
    preprocess = transforms.Compose([
       transforms.Resize(256),
       transforms.CenterCrop(224),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    classes = load_imagenet_classes()

    # Skip using detect_objects_in_fridge and classify the entire image
    item_label = classify_image(image, model, classes, preprocess)
    print(f"Full image classified as: {item_label}")

    try:
        chatgpt_response = query_chatgpt(image)
        print(f"ChatGPT API response for the full image:\n{chatgpt_response}\n")
    except Exception as e:
        print(f"Error querying ChatGPT for the full image: {e}")