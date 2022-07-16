# Módulo ServeInference

# Imports
import requests
import torch
from app import api
from PIL import Image, ImageFile
from io import BytesIO
from flask import jsonify
from flask_restful import Resource, reqparse
from preprocessing import preprocess_image
from tools import read_imagenet_classnames, one_prediction, run_inference, parse_base64

# Parâmetro global
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Argumentos
parser = reqparse.RequestParser(bundle_errors = True)
parser.add_argument('img_url', help = 'URL da imagem')
parser.add_argument('pred_num', type = int, default = 5, help = 'Número de previsões')

# Carrega o modelo
model = torch.jit.load("checkpoints/model.pt")

# Lista de classes
imagenet_classes = read_imagenet_classnames("cache/imagenet_classnames.txt")

# Classe do ServeInference
class ServeInference(Resource):

    # Método get
    def get(self):

        # Parse dos argumentos
        args = parser.parse_args()

        # URL da imagem
        img_url = args.img_url

        # Checa o número de previsões
        if args.pred_num:
            top_predictions = args.pred_num

        # Converte a imagem para a base64
        bytes_str = parse_base64(img_url)
        
        # Se estiver no formato correto
        if not bytes_str:

            # Request da imagem
            res = requests.get(img_url)

            # Status
            if res.status_code != 200:
                try:
                    res.raise_for_status()
                except Exception as e:
                    return jsonify({"status": res.status_code, "msg": str(e)})
            else:
                bytes_str = res.content
            
        # Abre a imagem
        im = Image.open(BytesIO(bytes_str))

        # Pré-processa a imagem
        im = preprocess_image(im)

        # Inferência
        prediction = run_inference(model, im, top_predictions)
        prediction = one_prediction(prediction, imagenet_classes)

        # Resposta
        response = {"status": 200, "msg": prediction}

        return jsonify(response)

# Cria o recurso no servidor
api.add_resource(ServeInference, '/app')


