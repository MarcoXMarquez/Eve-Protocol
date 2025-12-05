import sys
import os

# --- BLOQUE DE ARREGLO DE RUTAS (IMPORTANTE PARA EL EXE) ---
if getattr(sys, 'frozen', False):
    # Si estamos en el .EXE, la ruta base es la carpeta temporal _MEIPASS
    base_path = sys._MEIPASS
    # Añadimos _MEIPASS al path para que encuentre 'src' si está copiado ahí
    sys.path.append(base_path)
else:
    # Si estamos en modo desarrollo (VSCode/Terminal), la raíz es el directorio padre de 'src'
    current_dir = os.path.dirname(os.path.abspath(__file__)) # carpeta src/
    root_dir = os.path.dirname(current_dir) # carpeta EmotionRecognitionModel/
    sys.path.append(root_dir)
# -----------------------------------------------------------

import tkinter as tk
# Ahora sí funcionarán estos imports
from src.ui.window import EmotionApp 

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionApp(root)
    root.mainloop()