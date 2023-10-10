from PIL import Image
import base64
import io
import numpy as np

def decode_image(image_base64_encoded):
    base64_decoded = base64.b64decode(image_base64_encoded)
    image = Image.open(io.BytesIO(base64_decoded))
    return np.array(image)