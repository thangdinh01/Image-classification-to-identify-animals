import torch
from torchvision import transforms
from PIL import Image
import cv2
from src.test_model import load_model, get_class_names

camera_active = False

def predict_from_camera(data_dir=r"D:\Python\BTLXLYA\data_split\train", model_path="./src/animal_model.pth", threshold=0.7):
    global camera_active
    camera_active = True

    # Kiểm tra và load các lớp
    try:
        classes = get_class_names(data_dir)
    except Exception as e:
        print(f"Error loading classes: {e}")
        return

    # Load model
    try:
        model = load_model(model_path, num_classes=len(classes))
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Khởi động camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access camera.")
        return

    # Transform
    transform = transforms.Compose([
        lambda img: transforms.functional.resize(img, (300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    while cap.isOpened() and camera_active:
        ret, frame = cap.read()
        if not ret:
            break

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = transform(image).unsqueeze(0)

        # Dự đoán
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)

        label = "Unknown" if confidence.item() < threshold else classes[predicted.item()]
        cv2.putText(frame, f"Predicted: {label} ({confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Camera Inference', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or not camera_active:
            break

    cap.release()
    cv2.destroyAllWindows()


def stop_camera():
    global camera_active
    camera_active = False
    cv2.destroyAllWindows()
