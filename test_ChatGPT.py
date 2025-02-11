import os
from openai import OpenAI
import base64
import requests

API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=API_KEY)

if not API_KEY:
    raise ValueError("API key is missing! Set the OPENAI_API_KEY environment variable.")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Function to analyze fridge contents using OpenAI's vision model
def analyze_fridge(image_path):
    
    # Getting the Base64 string
    base64_image = encode_image(image_path)

    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Analyze the image and list all fridge items along with their quantities. "
                        "For each item, output exactly one line in the following format: "
                        "'ITEM: QUANTITY'. If an item does not have an associated quantity, use 'none' as the quantity. "
                        "Do not include any additional text, explanations, or formatting."
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        }
    ],
)
    print(response.choices[0].message.content)
    # Parse the response to create a dictionary
    items = {}
    for line in response.choices[0].message.content.split('\n'):
        if ": " in line:
            item, quantity = line.split(": ")
            items[item.strip()] = quantity.strip()
    
    return items

def add_to_dict(dictionary, key, value):
    if key in dictionary:
        dictionary[key] += value
    else:
        dictionary[key] = value

# Example usage
if __name__ == "__main__":
    image_path = "fridge_items_2.webp"  # Replace with your actual fridge image file path
    try:
        result = analyze_fridge(image_path)
    except Exception as e:
        print(f"Error: {e}")