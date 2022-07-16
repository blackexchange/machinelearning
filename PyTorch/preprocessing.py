# Módulo de pré-processamento dos dados

# Imports
import os
import torch
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms

# Regra de transformações das imagens de acordo com os requerimentos da arquitetura do modelo
transformations = transforms.Compose([transforms.Resize((256, 256)), 
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                           std = [0.229, 0.224, 0.225])])

# Função de pré-processamento de cada imagem
# Usaremos na API e para processar cada imagem de forma individual
def preprocess_image(img):

    # Abre a imagem e carrega na memória
    if type(img) is str:
        img = Image.open(img)

    # Aplica as transformações
    im = transformations(img)

    # Ajusta o formato do tensor
    im = torch.unsqueeze(im, 0)

    return im

# Função de pré-processamento dos dados
# Usaremos no módulo de execução do modelo para testar várias imagens em uma única execução
def preprocess_data(path: str):

    # Lista para os tensores
    data_tensors = []

    # Lista para os nomes dos arquivos
    file_names = []

    # Loop pela pasta com as imagens
    for file_path in os.listdir(path):

        # Separa nome de arquivo e extensão
        file_name, extension = os.path.splitext(file_path)

        # Se a extensão for válida, continua
        if extension not in [".jpg", ".png"]:
            continue
        
        # Concatena pasta e imagem
        file_path = os.path.join(path, file_path)

        # Cria a lista de tensores
        data_tensors.append(preprocess_image(file_path))

        # Cria a lista de nomes de arquivos
        file_names.append(file_name)

    # Ajusta o shape da lista de tensores
    data_tensors = torch.vstack(data_tensors)

    return data_tensors, file_names




