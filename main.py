from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.models import load_model
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)  # Permite requisições de outros domínios (como o frontend)

# Carregar o modelo e as classes
model = load_model("keras_Model.h5", compile=False)
class_names = ['pneumonia', 'normal']

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Ler a imagem enviada
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Processar a imagem
    image = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1

    # Fazer a predição
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return jsonify({
        "class": class_name,
        "confidence": float(confidence_score)
    })

# Executar a aplicação apenas no ambiente local
if __name__ == '__main__':
    app.run(debug=True)
