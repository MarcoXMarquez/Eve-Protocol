# Eve Protocol — Reconocimiento de Emociones (resumen)

Proyecto de reconocimiento de emociones faciales que agrupa scripts de
entrenamiento, evaluación e inferencia. Este `README` resume cómo usar
el código que se encuentra en el directorio `src/` y otros recursos
adjuntos (`data/`, `models_saved/`).

**Quick Start**
- **Instalar dependencias**: `pip install -r requirements.txt`
- **Activar entorno (Windows PowerShell)**: `env\Scripts\Activate.ps1`
- **Ejecutar inferencia de ejemplo**: `python .\src\main.py --image <ruta_imagen>`

**Comandos útiles**
- **Entrenar**: `python .\src\train.py`
- **Evaluar**: `python .\src\evaluate.py`
- **Visualizar/depurar resultados**: `python .\src\viewer.py`

**Archivos principales (en `src/`)**
- **`main.py`**: Punto de entrada para inferencia/servicio de detección
    e inferencia sobre imágenes (usa los servicios en `src/services`).
- **`train.py`**: Entrenamiento del modelo; carga datos desde
    `data/data_combined/` (o rutas configuradas) y guarda checkpoints.
- **`evaluate.py`**: Evaluación del modelo entrenado, genera métricas
    (accuracy, matriz de confusión, métricas por clase).
- **`viewer.py`**: Herramienta para mostrar resultados, imágenes y
    métricas de entrenamiento/evaluación.

**Módulos y responsabilidades**
- **`src/models/model.py`**: Definición de la arquitectura (PyTorch).
- **`src/utils/data_loader.py`**: Carga y preprocesado de datasets
    (transformaciones, batching, particiones train/val).
- **`src/services/detector.py`**: Detección de rostros y recorte de
    regiones de interés (usa OpenCV u otros detectores).
- **`src/services/goku_ai.py`**: Lógica de inferencia / orquestación
    entre detector y modelo (API interna del proyecto).
- **`src/ui/window.py`**: Interfaz mínima para mostrar resultados en
    ventana (si aplica).

**Datos y modelos**
- **Datasets**: `data/` contiene subcarpetas `raw/` y `data_combined/`
    con particiones `train/` y `val/` organizadas por clase.
- **Modelos guardados**: `models_saved/emotion_model.pth` (checkpoint
    ejemplo). Ajusta rutas en los scripts si utilizas otra ubicación.

**Configuración y recomendaciones**
- Revisa y adapta rutas dentro de `src/utils/data_loader.py` si tus
    datos no están en `data/data_combined/`.
- Para acelerar entrenamiento con GPU, instala la versión de PyTorch
    compatible con tu CUDA y sigue las instrucciones oficiales de
    `https://pytorch.org/`.

**Desarrollo**
- Usa `env` (ya incluida en el repo) o crea un virtualenv: `python -m venv env`.
- Formato y lint: aplica tus herramientas locales (`black`, `flake8`).

**Estructura del repositorio**
- **`src/`**: Código fuente (ver arriba para archivos clave).
- **`data/`**: Imágenes y datasets (`raw/`, `data_combined/`).
- **`models_saved/`**: Modelos entrenados y checkpoints.
- **`requirements.txt`**: Dependencias del proyecto.
