from flask import Flask, render_template, request, jsonify
import requests
import numpy as np
import cv2
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Function to fetch solar image from Helioviewer API
def fetch_solar_image(date):
    url = "https://api.helioviewer.org/v2/getJpeg/"
    params = {"date": date, "sourceId": 14, "width": 1024, "height": 1024}
    
    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            return BytesIO(response.content)
        else:
            return None
    except requests.exceptions.RequestException as e:
        print(f"API Request Error: {e}")
        return None

# Function to process the image with OpenCV
def process_image(image_data):
    image = Image.open(image_data)
    image_np = np.array(image)
    gray_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges_img = cv2.Canny(gray_img, 50, 150)
    return gray_img, edges_img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_image', methods=['POST'])
def get_image():
    date = request.form['date']
    image_data = fetch_solar_image(date) 
    
    if image_data is None:
        return jsonify({'error': 'Failed to fetch image'}), 500
    
    gray_img, edges_img = process_image(image_data)
    _, gray_encoded = cv2.imencode('.png', gray_img)
    _, edges_encoded = cv2.imencode('.png', edges_img)
    
    return jsonify({
        'gray_img': gray_encoded.tobytes().hex(),
        'edges_img': edges_encoded.tobytes().hex()
    })

if __name__ == '__main__':
    app.run(debug=True)
