import torch

# Verificar si la interfaz CUDA está disponible para PyTorch.
print("CUDA disponible:", torch.cuda.is_available())

# Obtener y mostrar el nombre de la GPU principal (índice 0).
# Este comando solo se ejecuta si CUDA está disponible.
if torch.cuda.is_available():
    print("Nombre GPU:", torch.cuda.get_device_name(0))
    # Obtener y mostrar la información de la memoria de la GPU (total y libre).
    # Retorna (memoria libre, memoria total) en bytes.
    print("Memoria disponible:", torch.cuda.mem_get_info())