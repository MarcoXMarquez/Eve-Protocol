import torch
import torch.nn as nn
import torchvision.models as models

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()

        # Carga del modelo base: ResNet18 preentrenado.
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Adaptar conv1 para aceptar entrada en escala de grises (1 canal).
        self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)


        # Definición del clasificador FC personalizado.
        self.base_model.fc = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            # Capa final para la clasificación de 'num_classes' emociones.
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Propagación de la entrada a través del modelo base modificado.
        return self.base_model(x)


# Verificación unitaria
if __name__ == '__main__':
    model = EmotionCNN(num_classes=7)
    print(model)

    # Simular entrada: lote (4) de imágenes 1x48x48.
    sample_input = torch.randn(4, 1, 48, 48)
    output = model(sample_input)
    print("Output shape:", output.shape) # Salida: [4, 7]