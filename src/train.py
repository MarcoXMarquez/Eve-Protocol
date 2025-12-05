import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import Counter
from models.model import EmotionCNN
from utils.data_loader import get_loaders
from tqdm import tqdm

def train_model(data_dir, epochs=500, batch_size=64, patience=7): 
    # Configuración del dispositivo de ejecución (GPU/CPU).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo detectado: {device}")

    # Verificación de la ruta del dataset.
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Error: La carpeta '{data_dir}' no existe. Asegúrese de que la ruta es correcta.")

    # Carga de datasets y DataLoaders.
    dataset_names = os.listdir(data_dir)
    print(f"Cargando datos desde '{data_dir}' con {len(dataset_names)} datasets...")
    for dataset_name in tqdm(dataset_names, desc="Cargando datasets", unit="dataset"):
        dataset_path = os.path.join(data_dir, dataset_name)
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Error: Dataset {dataset_name} no encontrado en {dataset_path}")

    # Obtener los DataLoaders balanceados y las clases.
    train_loader, val_loader, classes = get_loaders(data_dir, batch_size=batch_size)
    print(f"Datos cargados correctamente. Clases detectadas: {classes}")

    # Función auxiliar para verificar la distribución de etiquetas.
    def check_class_distribution(loader):
        all_labels = []
        for _, labels in loader:
            all_labels.extend(labels.numpy())
        return Counter(all_labels)

    # Mostrar la distribución de clases en el conjunto de entrenamiento.
    train_counter = check_class_distribution(train_loader)
    print("\nDistribución de clases en entrenamiento:")
    for emotion, count in zip(classes, train_counter.values()):
        print(f" - {emotion}: {count} muestras")

    # --- Cálculo de Pesos para la Función de Pérdida ---
    # Convertir conteos a tensores de flotante.
    class_counts = torch.tensor(list(train_counter.values()), dtype=torch.float)
    # Calcular el peso inverso (1/frecuencia) para mitigar el desequilibrio de clases.
    class_weights = 1. / class_counts
    # Normalizar los pesos para que sumen 1.
    class_weights = class_weights / class_weights.sum()

    print("\nConfigurando modelo y optimizador...")
    # Inicialización del modelo y envío al dispositivo.
    model = EmotionCNN().to(device)
    # Función de pérdida: CrossEntropyLoss con pesos de clase.
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    # Optimizador Adam con tasa de aprendizaje inicial y regularización L2 (weight_decay).
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    # Scheduler: ReduceLROnPlateau, reduce LR si la métrica no mejora.
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

    best_val_acc = 0.0
    no_improve = 0 # Contador para Early Stopping

    print("\nEntrenamiento iniciado...\n")

    # Bucle principal de entrenamiento.
    with tqdm(range(epochs), desc="Épocas", unit="epoch") as epoch_bar:
        for epoch in epoch_bar:
            # Modo entrenamiento.
            model.train()
            total_loss = 0
            correct_train = 0
            total_train = 0

            # Bucle de entrenamiento por lote.
            loop = tqdm(train_loader, desc=f"Época {epoch+1}/{epochs} (train)", leave=False)
            for images, labels in loop:
                # Mover datos al dispositivo.
                images, labels = images.to(device), labels.to(device)
                
                # Paso 1: Reiniciar gradientes.
                optimizer.zero_grad()
                # Paso 2: Pase hacia adelante (Forward Pass).
                outputs = model(images)
                # Paso 3: Calcular la pérdida.
                loss = criterion(outputs, labels)
                # Paso 4: Pase hacia atrás (Backpropagation).
                loss.backward()
                # Paso 5: Actualizar pesos.
                optimizer.step()

                # Métricas de entrenamiento.
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

                loop.set_postfix(loss=loss.item())

            train_acc = correct_train / total_train

            # --- Fase de Validación ---
            model.eval()
            correct_val, total_val, val_loss = 0, 0, 0
            with torch.no_grad(): # Desactivar gradientes en validación.
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            val_acc = correct_val / total_val
            avg_val_loss = val_loss / len(val_loader)

            # Actualizar la tasa de aprendizaje con el valor de 'val_acc'.
            scheduler.step(val_acc)

            # Actualizar la barra de progreso con las métricas de la época.
            epoch_bar.set_postfix(
                train_acc=f"{train_acc*100:.2f}%",
                val_acc=f"{val_acc*100:.2f}%",
                train_loss=f"{total_loss/len(train_loader):.4f}",
                val_loss=f"{avg_val_loss:.4f}"
            )
            
            # IMPRESION DEL RESUMEN DE LA ÉPOCA (AQUÍ ESTÁ EL CAMBIO CLAVE)
            print(f"| Época {epoch+1}/{epochs} | Acc. Entr.: {train_acc*100:.2f}% | Pérdida Entr.: {total_loss/len(train_loader):.4f} | Acc. Val.: {val_acc*100:.2f}% | Pérdida Val.: {avg_val_loss:.4f} |")

            # --- Lógica de Early Stopping y Guardado del Modelo ---
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve = 0
                # Guardar únicamente los pesos del modelo (state_dict) en la mejor iteración.
                torch.save(model.state_dict(), "emotion_model.pth")
                print(f"-> Mejor modelo guardado. Precisión de validación: {val_acc*100:.2f}%")
            else:
                no_improve += 1
                print(f"-> Sin mejora. Intentos sin mejora: {no_improve}/{patience}")
                if no_improve >= patience:
                    print("-> Criterio de 'Early Stopping' activado. Finalización del entrenamiento.")
                    break

    print(f"\nEntrenamiento finalizado. Mejor precisión en validación: {best_val_acc*100:.2f}%")

if __name__ == "__main__":
    # Asegúrate de que esta ruta sea correcta: data/raw/data_combined
    train_model("data/raw/data_combined")