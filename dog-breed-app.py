from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from PIL import Image

app = Flask(__name__)

# 모델 로드
model_path = 'models/dog_breed_model.h5'
model = tf.keras.models.load_model(model_path)

def load_dog_breeds(filepath):
    """ 파일에서 개 품종 목록을 로드한다. """
    with open(filepath, 'r') as file:
        dog_breeds = [line.strip() for line in file]
    return dog_breeds

# 'models/dogbreed_classes.txt'에서 DOGBREED_CLASSES 로드
DOGBREED_CLASSES = load_dog_breeds('models/dogbreed_classes.txt')
DOGBREED_CLASSES.sort()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided.'}), 400
    file = request.files['file']
    if not file:
        return jsonify({'error': 'No file provided.'}), 400

    # 이미지 파일 저장 위치 설정
    filename = 'uploaded_image.jpg'
    filepath = os.path.join('static', filename)
    file.save(filepath)

    # 이미지 읽기 및 전처리
    img = Image.open(filepath).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape((1, 224, 224, 3))
    
    # 예측 수행
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_probability = predictions[0][predicted_index]

    # 예외 처리를 통한 안전한 인덱스 접근
    try:
        predicted_breed = DOGBREED_CLASSES[predicted_index]
    except IndexError as e:
        # 로깅을 통한 오류 진단
        app.logger.error(f'IndexError: {e} - Predicted index out of range. \
                         Predicted Index: {predicted_index}, Labels Count: {len(DOGBREED_CLASSES)}')
        return jsonify({'error': 'Predicted index is out of range.'}), 500

    return render_template('index.html', predicted_breed=predicted_breed, \
                predicted_probability="{:.1%}".format(predicted_probability), image_path=filepath)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
