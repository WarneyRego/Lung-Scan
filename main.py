from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.models import load_model
import numpy as np
import cv2
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)
CORS(app)  # Permite requisições de outros domínios (como o frontend)
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173"]}})


# Verificar se o modelo existe antes de carregar
MODEL_PATH = "keras_model.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"O modelo '{MODEL_PATH}' não foi encontrado.")

# Carregar o modelo e definir as classes
model = load_model(MODEL_PATH, compile=False)
class_names = ['pneumonia', 'normal']

@app.route('/predict', methods=['POST'])
def predict():
    """
    Rota para fazer a predição com base em uma imagem enviada pelo cliente.
    """
    # Verificar se um arquivo foi enviado na requisição
    if 'file' not in request.files:
        return jsonify({"error": "Nenhum arquivo foi enviado."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Nenhum arquivo foi selecionado."}), 400

    try:
        # Ler o arquivo enviado e converter para um array NumPy
        npimg = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Não foi possível processar a imagem enviada."}), 400

        # Pré-processar a imagem (redimensionar e normalizar)
        image = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
        image = (image / 127.5) - 1  # Normalização para [-1, 1]

        # Fazer a predição
        prediction = model.predict(image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Retornar a classe predita e o nível de confiança
        return jsonify({
            "class": class_name,
            "confidence": float(confidence_score)
        })
    except Exception as e:
        return jsonify({"error": f"Ocorreu um erro ao processar a imagem: {str(e)}"}), 500


# Executar a aplicação no modo de desenvolvimento
if __name__ == '__main__':
    app.run(debug=True)
