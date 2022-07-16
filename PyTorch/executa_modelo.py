# Módulo para executar o modelo

# Obs: Este módulo NÃO é a API.

# Imports
import os
import argparse
import torch
from preprocessing import preprocess_data
from tools import read_imagenet_classnames, display_results, run_inference, parse_base64
from torchvision import models

# Argumentos
parser = argparse.ArgumentParser(description = 'Inferência com o Modelo Treinado')
parser.add_argument('-tp', '--top-predictions', metavar = 'NUMPRED', default = 5, help = 'Número de previsões por imagem')
 
# Executa o programa
if __name__ == "__main__":

    # Parse dos argumentos
    args = parser.parse_args()

    # Importa o modelo
    model = models.resnet18(pretrained = True)
    
    # Carrega os nomes das classes
    imagenet_classes = read_imagenet_classnames("cache/imagenet_classnames.txt")

    # Pasta de cache
    data = preprocess_data("cache")

    # Executa a inferência
    predictions = run_inference(model, data[0], int(args.top_predictions))

    # Mostra os resultados
    display_results(data, predictions, imagenet_classes)



