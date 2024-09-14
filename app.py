from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests
from flask import Flask, request, jsonify
import time

app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/ocr', methods=['POST'])
def ocr():
    url = request.json.get('url')
    image = Image.open(requests.get(url, stream=True).raw)
    start = time.time()
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    print(f"Model load time: {time.time() - start} seconds")
    start = time.time()
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    print(f"OCR time: {time.time() - start} seconds")
    return jsonify({
        'text': generated_texts
    })

if __name__ == '__main__':
    app.run(debug=True)