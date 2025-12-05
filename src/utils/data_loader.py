import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from collections import Counter

def get_loaders(data_root, batch_size=64, val_split=0.2):
    # Definición del pipeline de transformaciones para el conjunto de entrenamiento.
    train_transform = transforms.Compose([
        # Convertir imágenes a escala de grises (1 canal).
        transforms.Grayscale(num_output_channels=1),
        # Redimensionar las imágenes al tamaño de entrada del modelo.
        transforms.Resize((48, 48)),
        # Aplicar aumento de datos: volteo horizontal aleatorio.
        transforms.RandomHorizontalFlip(p=0.5),
        # Aplicar rotación aleatoria de hasta 15 grados.
        transforms.RandomRotation(15),
        # Aplicar traslación aleatoria (zoom/paneo).
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        # Ajuste aleatorio de brillo y contraste para robustez.
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        # Convertir la imagen a tensor de PyTorch.
        transforms.ToTensor(),
        # Normalizar el tensor (escala de -1 a 1) para el entrenamiento.
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # Definición del pipeline de transformaciones para el conjunto de validación (solo pasos necesarios).
    val_transform = transforms.Compose([
        # Convertir a escala de grises.
        transforms.Grayscale(num_output_channels=1),
        # Redimensionar la imagen.
        transforms.Resize((48, 48)),
        # Convertir a tensor de PyTorch.
        transforms.ToTensor(),
        # Normalizar el tensor.
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # Cargar los datasets utilizando la estructura de directorios de ImageFolder.
    train_data = datasets.ImageFolder(root=os.path.join(data_root, "train"), transform=train_transform)
    val_data = datasets.ImageFolder(root=os.path.join(data_root, "val"), transform=val_transform)

    # --- Configuración del Muestreo Ponderado (Clase Imbalance) ---
    # Contar la frecuencia de cada clase en el dataset de entrenamiento.
    class_counts = Counter([label for _, label in train_data])
    # Calcular el peso inverso (1/frecuencia) para cada muestra.
    class_weights = [1.0 / class_counts[label] for _, label in train_data]
    # Crear un 'Sampler' que elija las muestras basándose en los pesos calculados (balanceo).
    sampler = WeightedRandomSampler(class_weights, num_samples=len(class_weights), replacement=True)

    # Inicializar el DataLoader de entrenamiento utilizando el 'Sampler' para balanceo.
    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)
    # Inicializar el DataLoader de validación (sin muestreo ponderado, solo lectura).
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Obtener los nombres de las clases detectadas por ImageFolder.
    class_names = train_data.classes

    # Retornar los DataLoaders y los nombres de las clases.
    return train_loader, val_loader, class_names