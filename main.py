from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from paddleocr import PaddleOCR
import numpy as np
import cv2
import re



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the models
model = YOLO("/vehicles_and_plates.pt")
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def detect_read_number_plates(image: np.ndarray):
    global model
    global ocr

    # Perform number plate detection on the image
    results = model.predict(source=image)

    output_texts = []

    # Process the results
    for r in results:
        boxes = r.boxes
        best_box = None
        max_confidence = 0.0

        for box in boxes:
            confidence = box.conf.item()
            if confidence > 0.5 and confidence > max_confidence:
                max_confidence = confidence
                best_box = box

        if best_box is None:
            continue

        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        plate_crop = image[y1:y2, x1:x2]

        # Convert to RGB
        plate_crop = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)

        # Use PaddleOCR to read text from the cropped plate image
        ocr_result = ocr.predict(plate_crop)

        boxes = ocr_result[0]['rec_boxes']
        texts = ocr_result[0]['rec_texts']
        left_to_right = sorted(zip(boxes, texts), key=lambda x: min(x[0][::2]))

        whitelist_pattern = re.compile(r'^[A-Z0-9]+$')
        left_to_right = ''.join([t for _, t in left_to_right])
        output_text = ''.join([t for t in left_to_right if whitelist_pattern.fullmatch(t)])
        output_texts.append(output_text)

    return output_texts

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    # Read the image file
    image = await file.read()
    image = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Call the detection function
    results = detect_read_number_plates(image)
    return {"number_plate_texts": results}

@app.get("/")
async def read_root():
    return {"message": "Hello from FastAPI in Colab!"}
