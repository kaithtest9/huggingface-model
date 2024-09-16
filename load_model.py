from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests
import time

# load image from the IAM database
with open("20150624_135325.jpg", "rb") as f:
    image = Image.open(f).convert("RGB")

start = time.time()
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-handwritten')
pixel_values = processor(images=image, return_tensors="pt").pixel_values

generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("time taken: ", time.time() - start)

print("text generated: ", generated_text)