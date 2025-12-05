from groq import Groq
from gtts import gTTS 
import pygame
import os
import time

# --- CLAVE DE API ---
GROQ_API_KEY = "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" 

class EveService:
    def __init__(self):
        # Inicialización del cliente Groq para el acceso al LLM.
        try:
            self.groq_client = Groq(api_key=GROQ_API_KEY)
        except:
            # Manejo de error si la clave API es inválida o falta.
            self.groq_client = None
        
        # Inicializar el módulo de audio de Pygame.
        try:
            pygame.mixer.init()
        except:
            pass # Si falla la inicialización del mixer, el servicio de audio no estará disponible.

    def get_eve_text(self, emotion):
        if not self.groq_client: return "Error de API Key."
        
        # Definición del prompt para el LLM. Se especifican el rol, el límite de longitud y el objetivo.
        prompt = f"""
        Eres un robot de ayuda. Tu amigo siente la emoción: '{emotion}'.
        Responde corto (máx 20 palabras). 
        Usa un lenguaje amigable para que se sienta mejor si es emocion negativa, 
        o si estan con una emocion positiva ayudalos a seguir asi.
        """
        try:
            # Llamada a la API de Groq utilizando el modelo especificado.
            resp = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile" 
            )
            # Extracción del contenido de la respuesta del LLM.
            return resp.choices[0].message.content
        except Exception as e:
            print(f"Error Groq: {e}")
            # Mensaje de fallback en caso de fallo en la comunicación con la API.
            return "¡Hola! Vamos a entrenar para que te sientas mejor."

    def generate_and_play_audio(self, text, callback_start=None, callback_end=None):
        """
        Genera el archivo de audio (MP3) a partir del texto y lo reproduce.
        Utiliza callbacks para sincronizar animaciones externas con el inicio/fin de la reproducción.
        """
        if not text: return

        # Nombre del archivo temporal de audio.
        filename = "eve_voz.mp3"
        
        try:
            # 1. Creación del audio utilizando gTTS (idioma español, acento latinoamericano).
            tts = gTTS(text=text, lang='es', tld='com.mx')
            tts.save(filename)
            
            # 2. Reproducción del audio con Pygame.
            pygame.mixer.music.load(filename)
            
            # Ejecutar callback para indicar el inicio de la animación de voz.
            if callback_start: callback_start()
            
            pygame.mixer.music.play()
            
            # Bucle de espera que mantiene el programa activo hasta que el audio termine.
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10) # Chequeo periódico del estado del mixer.
            
            # Ejecutar callback para indicar el fin de la animación de voz.
            if callback_end: callback_end()

            # 3. Limpieza del archivo temporal.
            pygame.mixer.music.unload()
            try:
                os.remove(filename)
            except:
                pass # Ignorar errores si el archivo está bloqueado temporalmente.
                
        except Exception as e:
            print(f"Error audio: {e}")
            # Asegurar la detención de la animación si ocurre un fallo durante la reproducción.
            if callback_end: callback_end()