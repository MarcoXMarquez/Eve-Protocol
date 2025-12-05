import torch
import cv2
from torchvision import transforms
from PIL import Image
import os
import sys

# Ajuste del PATH para importar el módulo del modelo ('EmotionCNN') en modo desarrollo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.models.model import EmotionCNN 

def resource_path(relative_path):
    """ Obtiene la ruta absoluta al recurso, funciona para dev y para PyInstaller """
    try:
        # PyInstaller crea una carpeta temporal en _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class EmotionDetector:
    def __init__(self):
        # Definición de las clases de emoción
        self.classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.device = torch.device('cpu')
        
        # --- CARGA DEL MODELO ---
        # Usamos resource_path para buscar el modelo dentro del .exe o en la carpeta local
        model_path = resource_path(os.path.join('models_saved', 'emotion_model.pth'))

        self.model = self._load_model(model_path)

        # --- CARGA DEL HAAR CASCADE ---
        # Intentamos cargar el xml local (si lo empaquetaste con --add-data), si no, usa el de sistema
        local_cascade = resource_path("haarcascade_frontalface_default.xml")
        
        if os.path.exists(local_cascade):
            self.face_cascade = cv2.CascadeClassifier(local_cascade)
        else:
            # Fallback al de la librería (puede fallar en .exe si no se configura bien)
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def _load_model(self, path):
        if not os.path.exists(path):
            print(f"ERROR CRÍTICO: No encuentro el modelo en: {path}")
            return None
        
        try:
            model = EmotionCNN()
            model.load_state_dict(torch.load(path, map_location=self.device))
            model.eval()
            print(f"Modelo cargado desde: {path}")
            return model
        except Exception as e:
            print(f"Error cargando los pesos del modelo: {e}")
            return None

    def detect_face_and_emotion(self, image_path):
        if not self.model:
            return "Error Modelo", None, None

        img_cv = cv2.imread(image_path)
        if img_cv is None: return None, None, None

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            return None, None, None

        (x, y, w, h) = faces[0]
        face_img = img_cv[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)

        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
        ])

        input_tensor = transform(face_pil).unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_tensor)
            _, pred = torch.max(output, 1)

        emotion = self.classes[pred.item()]
        return emotion, (x, y, w, h), face_pil