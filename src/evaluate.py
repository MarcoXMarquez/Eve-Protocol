import torch
from sklearn.metrics import classification_report
from models.model import EmotionCNN
from utils.data_loader import get_loaders

def evaluate_model(data_dir):
    # Determinar el dispositivo de cómputo disponible (CUDA si existe, sino CPU).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Obtener los DataLoaders de entrenamiento y validación, y los nombres de las clases.
    train_loader, val_loader, classes = get_loaders(data_dir)
    
    # Inicializar la arquitectura del modelo y moverla al dispositivo seleccionado.
    model = EmotionCNN().to(device)
    
    # Cargar los pesos pre-entrenados del modelo desde el archivo PTH.
    model.load_state_dict(torch.load("emotion_model.pth", map_location=device))
    
    # Establecer el modelo en modo de evaluación (desactiva Dropout y BatchNorm).
    model.eval()

    # Listas para almacenar las predicciones y etiquetas verdaderas.
    all_preds = []
    all_labels = []

    # Deshabilitar el cálculo de gradientes para optimizar la inferencia.
    with torch.no_grad():
        # Iterar sobre el DataLoader de validación.
        for images, labels in val_loader:
            # Mover datos al dispositivo.
            images, labels = images.to(device), labels.to(device)
            # Pase hacia adelante (Forward Pass).
            outputs = model(images)
            # Obtener el índice de la clase con la máxima probabilidad (predicción).
            _, preds = torch.max(outputs, 1)
            
            # Almacenar las predicciones y etiquetas, moviéndolas de vuelta a la CPU.
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Imprimir el informe de clasificación (precisión, recall, f1-score) utilizando scikit-learn.
    print(classification_report(all_labels, all_preds, target_names=classes))


if __name__ == "__main__":
    # Ejecutar la función de evaluación con el directorio de datos.
    evaluate_model("data/raw/data_combined")