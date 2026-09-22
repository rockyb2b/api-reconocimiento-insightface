from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import cv2
import numpy as np
import os
import configparser
import logging
import sys
import uvicorn
from insightface.app import FaceAnalysis

app = FastAPI()

def get_application_path() -> str:
    """Return directory path depending on PyInstaller frozen state."""
    return os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))

def setup_logging(log_file: str = 'ApiReconocimientoInsightFace.log'):
    application_path = get_application_path()
    log_path = os.path.join(application_path, log_file)
    logging.basicConfig(
        filename = log_path,
        level = logging.INFO,
        format = '%(asctime)s %(levelname)s : %(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S'
    )
    logging.info(f"Logging initialized at {log_path}")
setup_logging()

# Umbral de similitud (coseno). Ajustable segun necesidad.
# Valor tipico: 0.35 - 0.45. Mas alto = mas estricto.
THRESHOLD = 0.40

# Nombre del archivo de cache de embedding
EMBEDDING_FILE = "embedding.npy"

# Instancia global del motor de reconocimiento (se carga una sola vez)
face_app = None

class UploadRequest(BaseModel):
    directorio_fotos: str
    image: str

class RegisterRequest(BaseModel):
    directorio_fotos: str
    image: str

def get_model_root():
    if getattr(sys, 'frozen', False):
        # Running inside PyInstaller
        base_path = sys._MEIPASS
    else:
        # Running normally
        base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, 'insightface')

def init_face_engine():
    """Inicializa el motor de InsightFace una sola vez al iniciar el servidor."""
    global face_app
    try:
        model_root = get_model_root()
        logging.info(f"Using model root: {model_root}")
        logging.info("Cargando modelo InsightFace (buffalo_l)...")
        face_app = FaceAnalysis(
                                    name='buffalo_l', 
                                    # root=app_path,   # 👈 THIS IS THE KEY , previous
                                    root=model_root,
                                    providers=['CPUExecutionProvider']
                                )
        face_app.prepare(ctx_id=0, det_size=(640, 640))
        logging.info("Modelo InsightFace cargado correctamente.")
    except Exception as e:
        logging.error(f"Error cargando modelo InsightFace: {str(e)}")
        raise


def normalize_image_size(img, target=640):
    """Redimensiona la imagen para que su lado mayor sea 'target' pixeles."""
    if img is None:
        return None
    h, w = img.shape[:2]
    max_side = max(h, w)
    if max_side == target:
        return img
    scale = target / max_side
    new_w = int(w * scale)
    new_h = int(h * scale)
    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(img, (new_w, new_h), interpolation=interpolation)


def get_embedding_from_image(img):
    """Extrae el embedding de la primera cara detectada en una imagen."""
    img = normalize_image_size(img)
    faces = face_app.get(img)
    if len(faces) == 0:
        return None
    return faces[0].normed_embedding

def cosine_similarity(emb1, emb2):
    """Similitud coseno entre dos embeddings ya normalizados."""
    return float(np.dot(emb1, emb2))


@app.post("/test")
async def ping():
    application_path = get_application_path()
    logging.info("Test endpoint was pinged.")
    return {"message": f"test ok, application_path = {application_path}"}

@app.post("/register")
async def register(request: RegisterRequest):
    """Registra el rostro de un empleado:
    - Guarda la foto recibida como 0_{doi}.jpg
    - Calcula el embedding y lo guarda como embedding.npy
    """
    try:
        directorio_fotos = request.directorio_fotos
        if not os.path.exists(directorio_fotos):
            os.makedirs(directorio_fotos, exist_ok=True)

        img_data = base64.b64decode(request.image)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Error reading image")

        # Normalizar y detectar rostro
        img_norm = normalize_image_size(img)
        faces = face_app.get(img_norm)

        if len(faces) == 0:
            raise HTTPException(status_code=400, detail="No se detecto rostro en la imagen")

        embedding = faces[0].normed_embedding

        # Guardar embedding cacheado
        cache_path = os.path.join(directorio_fotos, EMBEDDING_FILE)
        np.save(cache_path, embedding)

        # Guardar foto normalizada para referencia (JPEG calidad 92)
        # Buscar el nombre base a partir del directorio (ej: ".../rostros/12345678/")
        folder_name = os.path.basename(os.path.normpath(directorio_fotos))
        jpg_path = os.path.join(directorio_fotos, f"0_{folder_name}.jpg")
        cv2.imwrite(jpg_path, img_norm, [cv2.IMWRITE_JPEG_QUALITY, 92])

        logging.info(f"Registro OK en {directorio_fotos}")
        return {"success": True, "message": "Registro exitoso"}

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error en register: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload(request: UploadRequest):
    """Valida una foto contra el embedding cacheado de un empleado.
    Solo usa embedding.npy. Si no existe, responde "No encontrado".
    """
    try:
        directorio_fotos = request.directorio_fotos

        if not os.path.exists(directorio_fotos):
            logging.info(f"La carpeta '{directorio_fotos}' no existe.")
            return [{"name": "No encontrado", "distance": 0.0, "percentage": 0.0}]

        # Decodificar imagen recibida
        image_base64string = request.image
        img_data = base64.b64decode(image_base64string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            logging.error("Error reading image")
            raise HTTPException(status_code=400, detail="Error reading image")

        # Cargar embedding.npy. Si no existe pero hay JPG guardado,
        # regenerarlo automaticamente (self-healing).
        cache_path = os.path.join(directorio_fotos, EMBEDDING_FILE)

        if not os.path.isfile(cache_path):
            # Fallback: intentar regenerar desde la imagen JPG guardada
            folder_name = os.path.basename(os.path.normpath(directorio_fotos))
            jpg_path = os.path.join(directorio_fotos, f"0_{folder_name}.jpg")

            if os.path.isfile(jpg_path):
                logging.info(f"embedding.npy no existe, regenerando desde {jpg_path}")
                try:
                    ref_img = cv2.imread(jpg_path)
                    if ref_img is not None:
                        ref_img = normalize_image_size(ref_img)
                        ref_faces = face_app.get(ref_img)
                        if len(ref_faces) > 0:
                            regenerated = ref_faces[0].normed_embedding
                            np.save(cache_path, regenerated)
                            logging.info(f"embedding.npy regenerado OK en {directorio_fotos}")
                        else:
                            logging.warning(f"No se detecto rostro en {jpg_path}, no se pudo regenerar .npy")
                            return [{"name": "No encontrado", "distance": 0.0, "percentage": 0.0}]
                    else:
                        logging.error(f"No se pudo leer {jpg_path}")
                        return [{"name": "No encontrado", "distance": 0.0, "percentage": 0.0}]
                except Exception as e:
                    logging.error(f"Error regenerando embedding desde JPG: {e}")
                    return [{"name": "No encontrado", "distance": 0.0, "percentage": 0.0}]
            else:
                logging.info(f"No hay embedding.npy ni 0_{folder_name}.jpg en {directorio_fotos}")
                return [{"name": "No encontrado", "distance": 0.0, "percentage": 0.0}]

        try:
            known_embedding = np.load(cache_path)
        except Exception as e:
            logging.error(f"Error leyendo embedding.npy: {e}")
            return [{"name": "No encontrado", "distance": 0.0, "percentage": 0.0}]

        # Normalizar y detectar rostro en imagen recibida
        img = normalize_image_size(img)
        faces = face_app.get(img)

        matches = []
        for face in faces:
            query_embedding = face.normed_embedding
            sim = cosine_similarity(query_embedding, known_embedding)
            distance = float(1 - sim)
            percentage = float(sim * 100)

            folder_name = os.path.basename(os.path.normpath(directorio_fotos))
            name = folder_name if sim >= THRESHOLD else "No encontrado"

            if sim < THRESHOLD:
                logging.info(f"No Coincidencia en {directorio_fotos} - distance : {distance} , percentage : {percentage}")

            matches.append({"name": name, "distance": distance, "percentage": percentage})

        if len(matches) == 0:
            matches.append({"name": "No encontrado", "distance": 0.0, "percentage": 0.0})

        return matches
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Exception occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def get_port():
    config = configparser.ConfigParser()
    port = 5000
    try:
        application_path = get_application_path()
        config_path = os.path.join(application_path, 'config.ini')
        if os.path.exists(config_path):
            config.read(config_path)
            try:
                port = int(config['server']['port'])
            except (KeyError, ValueError):
                logging.warning(f"Port not found in config file or invalid. Using default port {port}")
        else:
            logging.warning(f"Config file not found. Using default port {port}")
    except Exception as e:
        logging.error(f"Exception occurred while reading config: {str(e)}")
    return port


if __name__ == '__main__':
    port = get_port()
    try:
        logging.info("---------------------------------------------------------------")
        logging.info(f"INICIANDO ApiReconocimiento en puerto : {port}")
        init_face_engine()
        # uvicorn.run(app, host='0.0.0.0', port=port, log_config=uvicorn_config)
        uvicorn.run(app, host='0.0.0.0', port=port, log_config=None)
    except Exception as e:
        print(f"Error : {str(e)}")
        logging.info(f"Error : {str(e)}")