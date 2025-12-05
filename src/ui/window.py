import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageSequence
import cv2
import threading
import os
import sys

# Importaciones internas
from src.services.detector import EmotionDetector
from src.services.goku_ai import EveService 

def resource_path(relative_path):
    """ Obtiene la ruta absoluta al recurso, funciona para dev y para PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# --- CLASE DE GESTIÓN DE ANIMACIONES GIF ---
class AnimatedGIF(tk.Label):
    def __init__(self, master, filename, delay=50): 
        self.master = master
        self.filename = filename
        self.delay = delay
        self.frames = []
        self.idx = 0
        self.animating = False
        
        try:
            # filename ya viene procesado por resource_path desde la llamada
            im = Image.open(filename)
            for frame in ImageSequence.Iterator(im):
                frame = frame.convert("RGBA")
                frame.thumbnail((450, 550), Image.Resampling.LANCZOS)
                self.frames.append(ImageTk.PhotoImage(frame))
        except Exception as e:
            print(f"Error cargando GIF {filename}: {e}")
            self.frames.append(ImageTk.PhotoImage(Image.new('RGB', (450, 550), 'black')))

        super().__init__(master, image=self.frames[0], bg="#050510", bd=0)

    def animate(self):
        if not self.animating: return
        self.idx = (self.idx + 1) % len(self.frames)
        self.config(image=self.frames[self.idx])
        self.after(self.delay, self.animate)

    def start(self):
        if not self.animating:
            self.animating = True
            self.animate()

    def stop(self):
        self.animating = False

# --- APLICACIÓN PRINCIPAL (TKINTER) ---
class EmotionApp:
    def __init__(self, root):
        self.root = root
        root.title("EVE - SYSTEM V.4.0")
        root.geometry("1100x750")
        root.configure(bg="#050510")

        self.detector = EmotionDetector()
        self.eve = EveService()
        self.image_path = None

        self._setup_ui()

    def _setup_ui(self):
        tk.Label(self.root, text=" ◈ EVE PROTOCOL ◈ ", font=("Consolas", 24, "bold"), 
                 bg="#050510", fg="#00ffcc").pack(pady=20)

        main_frame = tk.Frame(self.root, bg="#050510")
        main_frame.pack(expand=True, fill="both")

        # --- IZQUIERDA (CÁMARA/IMAGEN) ---
        left_frame = tk.Frame(main_frame, bg="#050510")
        left_frame.pack(side="left", padx=30, expand=True)

        self.img_border = tk.Frame(left_frame, bg="#00ffcc", padx=2, pady=2)
        self.img_border.pack()
        
        self.lbl_user_img = tk.Label(self.img_border, text="[ NO SIGNAL ]", 
                                     bg="#001100", fg="#005500", 
                                     width=40, height=20, font=("Consolas", 12))
        self.lbl_user_img.pack()

        # --- DERECHA (AVATAR) ---
        self.right_frame = tk.Frame(main_frame, bg="#050510")
        self.right_frame.pack(side="right", padx=30, expand=True)

        # Usamos resource_path para cargar los GIFs correctamente en el .EXE
        idle_path = resource_path("robot_quieto.gif")
        talk_path = resource_path("robot_hablando.gif")

        self.gif_idle = AnimatedGIF(self.right_frame, idle_path, delay=40)
        self.gif_talk = AnimatedGIF(self.right_frame, talk_path, delay=30) 

        self.gif_idle.pack()
        self.gif_idle.start()

        self.lbl_ia_text = tk.Label(self.right_frame, text="Esperando datos biométricos...", 
                                     font=("Courier New", 12, "bold"), wraplength=350, justify="center",
                                     bg="#050510", fg="#00ffcc", pady=20)
        self.lbl_ia_text.pack()

        # --- BOTONES ---
        btn_frame = tk.Frame(self.root, bg="#050510")
        btn_frame.pack(pady=40, side="bottom")

        btn_conf = {
            "font": ("Consolas", 12, "bold"), "bg": "#002233", "fg": "#00ffcc", 
            "activebackground": "#004455", "activeforeground": "white", 
            "bd": 2, "relief": "groove", "width": 18
        }

        tk.Button(btn_frame, text="📁 CARGAR IMAGEN", command=self.upload, **btn_conf).grid(row=0, column=0, padx=15)
        tk.Button(btn_frame, text="📹 CÁMARA", command=self.camera, **btn_conf).grid(row=0, column=1, padx=15)
        
        btn_analyze = btn_conf.copy()
        btn_analyze["bg"] = "#004400"
        btn_analyze["fg"] = "#00ff00"
        tk.Button(btn_frame, text="▶ INICIAR ESCANEO", command=self.process, **btn_analyze).grid(row=0, column=2, padx=15)

    def toggle_avatar(self, is_talking):
        if is_talking:
            self.gif_idle.pack_forget() 
            self.gif_idle.stop()
            self.gif_talk.pack(before=self.lbl_ia_text) 
            self.gif_talk.start()
        else:
            self.gif_talk.pack_forget() 
            self.gif_talk.stop()
            self.gif_idle.pack(before=self.lbl_ia_text) 
            self.gif_idle.start()

    def upload(self):
        path = filedialog.askopenfilename(filetypes=[("Img", "*.jpg *.png *.jpeg")])
        if path:
            self.image_path = path
            self.show_image(path)

    def show_image(self, path, coords=None):
        img = Image.open(path).convert("RGB")
        if coords:
            draw = ImageDraw.Draw(img)
            x, y, w, h = coords
            draw.rectangle([x, y, x+w, y+h], outline="#00ffcc", width=3)
            len_line = int(w/4)
            draw.line([(x,y), (x+len_line, y)], fill="#00ffcc", width=6)
            draw.line([(x,y), (x, y+len_line)], fill="#00ffcc", width=6)

        img.thumbnail((450, 450))
        self.photo_user = ImageTk.PhotoImage(img)
        self.lbl_user_img.config(image=self.photo_user, width=0, height=0)

    def process(self):
        if not self.image_path: return
        
        emotion, coords, _ = self.detector.detect_face_and_emotion(self.image_path)
        
        if not emotion:
            self.lbl_ia_text.config(text="ERROR: Sujeto no identificado.")
            return
            
        self.show_image(self.image_path, coords)
        threading.Thread(target=self.run_ai, args=(emotion,)).start()

    def run_ai(self, emotion):
        self.lbl_ia_text.config(text="ANALIZANDO PATRONES EMOCIONALES...", fg="yellow")
        
        text = self.eve.get_eve_text(emotion)
        self.lbl_ia_text.config(text=f'"{text}"', fg="#00ffcc")
        
        self.eve.generate_and_play_audio(
            text,
            callback_start=lambda: self.toggle_avatar(True),
            callback_end=lambda: self.toggle_avatar(False)
        )

    def camera(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened(): return
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            frame[:,:,0] = cv2.multiply(frame[:,:,0], 0.5) 
            frame[:,:,2] = cv2.multiply(frame[:,:,2], 0.5)
            
            cv2.imshow("SCANNER [Presiona S]", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('s'):
                cv2.imwrite("captura.jpg", frame)
                self.image_path = "captura.jpg"
                self.show_image("captura.jpg")
                break
                
        cap.release()
        cv2.destroyAllWindows()