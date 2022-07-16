# Módulo de ferramentas auxiliares

# Imports
import base64
import re
import torch.nn.functional as F

# Função para ler os nomes das classes
def read_imagenet_classnames(path:str):

    # Abre o arquivo para leitura  de cada linha
    with open(path, "r") as f:
        temp = f.readlines()

    # Strip dos caracteres
    temp = [i.strip(" ").strip("\n").strip(", ").split(":") for i in temp]
    temp = {int(k):v.strip(" ").split(",") for k,v in temp}

    # Lista de classes
    classes = []

    # Loop
    for i in temp:
        classes.append([k.strip(" ") for k in temp[i]])

    return classes

# Função para mostrar os resultados
def display_results(data, predictions, imagenet_classes, print_values = True):

    # Carrega os dados
    input_data, file_names = data

    # Carrega as previsões
    probabilities, pred_indices = predictions

    # Gera erro em caso de problema
    assert len(input_data) == len(file_names) == len(probabilities) == len(pred_indices), "Checando inconsistências"
    
    # Dicionário para as saídas
    pred_outputs = {}

    # Loop
    for i, file in enumerate(file_names):
        prob, idx = probabilities[i], pred_indices[i]
        prediction = [f"{imagenet_classes[idx[j]][0]} com probabilidade de {prob[j]:.2f}%" for j in range(len(prob))]
        prediction = "\n".join(prediction)
        
        if print_values:
            print("\n-------------------------------------")
            print("\nFazendo Previsão...")
            print(f"\n Arquivo {file} previsão: {prediction}")
        else:
            pred_outputs[file] = prediction
            if i == len(file_names) - 1:
                return pred_outputs

# Função para uma previsão
def one_prediction(prediction, imagenet_classes):

    # Previsões e índices
    prob, idx = prediction[0][0], prediction[1][0]

    return  [{"classe": str(imagenet_classes[idx[j]][0]).strip("'"), "probabilidade": str(prob[j]).strip("'")} for j in range(len(prob))]

# Função para a inferência
def run_inference(model, input_data, top_predictions):

    # Previsões
    predictions = model(input_data)

    # Extrai as probabilidades das previsões
    probabilities, pred_indices = F.softmax(predictions, 1).topk(top_predictions)

    # Multiplica as previsões de probabilidade por 100 no formato numpy
    probabilities = (probabilities * 100).detach().numpy()

    # Índices das previsões no formato numpy
    pred_indices = pred_indices.detach().numpy()
    
    return probabilities, pred_indices

# Parse da imagem na base 64
def parse_base64(string_):
    base64_path = "data:image/jpeg;base64,"
    if string_.startswith(base64_path):
        string_ = re.sub(base64_path, "", string_)
        string_ =  bytes(string_, "UTF-8")
        return base64.b64decode(string_)
    else:
        return None
