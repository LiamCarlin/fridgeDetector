import cv2
import numpy as np
from ultralytics import YOLO
import os
from openai import OpenAI
from PIL import Image
import io
import sys

# Initialize OpenAI client
client = OpenAI(
    api_key=""
)

# Initialize YOLO model
model = YOLO("yolov8n.pt")

search_cams = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def list_available_cameras():
    """List all available camera devices and return them in a dictionary."""
    camera_dict = {}
    for i in search_cams:  # Check first 10 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Try to get camera name/description (may not work on all systems)
            ret, frame = cap.read()
            if ret:
                resolution = f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
                camera_dict[i] = f"Camera {i} ({resolution})"
            else:
                camera_dict[i] = f"Camera {i}"
        cap.release()
    return camera_dict


def capture_fridge(camera_index=0):
    # Try to initialize camera
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        print("Available cameras:")
        available_cameras = list_available_cameras()
        if not available_cameras:
            print("No cameras found!")
            return
        print(f"Try running with a different camera index: {available_cameras}")
        return

    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Error: Could not read frame")
                break

            # Check if frame is empty
            if frame.size == 0:
                print("Error: Empty frame received")
                continue

            # Run YOLO detection
            results = model(frame)

            # Process each detection
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Crop the detected object
                    cropped = frame[y1:y2, x1:x2]

                    # Convert to PIL Image and then to bytes
                    pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                    img_byte_arr = io.BytesIO()
                    pil_img.save(img_byte_arr, format="JPEG")
                    img_byte_arr = img_byte_arr.getvalue()

                    # Get OpenAI description
                    response = client.chat.completions.create(
                        model="gpt-4-vision-preview",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "What food item is this?"},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{img_byte_arr}"
                                        },
                                    },
                                ],
                            }
                        ],
                        max_tokens=50,
                    )

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Add label
                    food_label = response.choices[0].message.content
                    cv2.putText(
                        frame,
                        food_label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

            # Display the frame
            cv2.imshow("Fridge Detection", frame)

            # Break loop with 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nStopping camera feed...")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cameras = list_available_cameras()

    if not cameras:
        print(
            "No cameras found! Please ensure your camera is connected and not in use by another application."
        )
        sys.exit(1)

    print("\nAvailable cameras:")
    for idx, desc in cameras.items():
        print(f"{idx}: {desc}")

    if len(sys.argv) > 1:
        camera_index = int(sys.argv[1])
        if camera_index not in cameras:
            print(f"Error: Camera index {camera_index} is not available")
            sys.exit(1)
    else:
        # If only one camera is available, use it automatically
        if len(cameras) == 1:
            camera_index = list(cameras.keys())[0]
            print(f"\nUsing the only available camera (index {camera_index})")
        else:
            # Ask user to select a camera
            while True:
                try:
                    camera_index = int(
                        input("\nSelect camera index from the list above: ")
                    )
                    if camera_index in cameras:
                        break
                    print("Invalid camera index, please try again")
                except ValueError:
                    print("Please enter a valid number")

    print(f"\nStarting camera {camera_index}...")
    print("Press 'q' to quit")
    capture_fridge(camera_index)
