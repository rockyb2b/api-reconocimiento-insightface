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


# ===========================================================
# CONFIGURACION GLOBAL desde config.ini
# Todos los parametros tienen valores default si no estan en el archivo
# ===========================================================

def load_config():
    """Carga config.ini con valores por default si no existe o falta algo."""
    config = configparser.ConfigParser()
    cfg = {
        'port': 5000,
        'host': '0.0.0.0',
        'threshold': 0.35,
        'det_size': 640,
        'log_level': 'INFO',
    }
    try:
        application_path = get_application_path()
        config_path = os.path.join(application_path, 'config.ini')
        if os.path.exists(config_path):
            config.read(config_path)

            if config.has_section('server'):
                try:
                    cfg['port'] = int(config['server'].get('port', cfg['port']))
                except (ValueError, KeyError):
                    pass
                cfg['host'] = config['server'].get('host', cfg['host'])

            if config.has_section('recognition'):
                try:
                    cfg['threshold'] = float(config['recognition'].get('threshold', cfg['threshold']))
                except (ValueError, KeyError):
                    pass
                try:
                    cfg['det_size'] = int(config['recognition'].get('det_size', cfg['det_size']))
                except (ValueError, KeyError):
                    pass

            if config.has_section('logging'):
                cfg['log_level'] = config['logging'].get('log_level', cfg['log_level']).upper()
    except Exception as e:
        print(f"Error leyendo config.ini, usando defaults: {e}")
    return cfg


CONFIG = load_config()


def setup_logging(log_file: str = 'ApiReconocimientoInsightFace.log'):
    application_path = get_application_path()
    log_path = os.path.join(application_path, log_file)
    level_map = {'DEBUG': logging.DEBUG, 'INFO': logging.INFO,
                 'WARNING': logging.WARNING, 'ERROR': logging.ERROR}
    level = level_map.get(CONFIG['log_level'], logging.INFO)
    logging.basicConfig(
        filename=log_path,
        level=level,
        format='%(asctime)s %(levelname)s : %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info(f"Logging initialized at {log_path}, level={CONFIG['log_level']}")


setup_logging()
logging.info(f"Config cargado: {CONFIG}")

EMBEDDING_FILE = "embedding.npy"
face_app = None


class UploadRequest(BaseModel):
    directorio_fotos: str
    image: str


class RegisterRequest(BaseModel):
    directorio_fotos: str
    image: str


def get_model_root():
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, 'insightface')


def init_face_engine():
    """Inicializa el motor de InsightFace una sola vez al iniciar el servidor."""
    global face_app
    try:
        model_root = get_model_root()
        det_size = CONFIG['det_size']
        logging.info(f"Using model root: {model_root}")
        logging.info(f"Cargando modelo InsightFace (buffalo_l) con det_size={det_size}...")
        face_app = FaceAnalysis(
            name='buffalo_l',
            root=model_root,
            providers=['CPUExecutionProvider']
        )
        face_app.prepare(ctx_id=0, det_size=(det_size, det_size))
        logging.info("Modelo InsightFace cargado correctamente.")
    except Exception as e:
        logging.error(f"Error cargando modelo InsightFace: {str(e)}")
        raise


def normalize_image_size(img, target=640):
    """Normaliza tamano de imagen para InsightFace."""
    if img is None:
        return None
    h, w = img.shape[:2]
    max_side = max(h, w)

    if max_side < target:
        scale = target / max_side
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    if max_side <= target * 1.5:
        return img

    scale = target / max_side
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def add_padding(img, pad_ratio=0.5):
    """Agrega padding gris alrededor de la imagen."""
    if img is None:
        return None
    h, w = img.shape[:2]
    pad_h = int(h * pad_ratio)
    pad_w = int(w * pad_ratio)
    return cv2.copyMakeBorder(
        img, pad_h, pad_h, pad_w, pad_w,
        cv2.BORDER_CONSTANT, value=[128, 128, 128]
    )


def get_best_face(faces):
    """De varios rostros detectados, devuelve el de mayor area (el mas cercano)."""
    if not faces or len(faces) == 0:
        return None
    if len(faces) == 1:
        return faces[0]
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def get_embedding_from_image(img):
    """Extrae el embedding del rostro principal en una imagen."""
    img = normalize_image_size(img)
    faces = face_app.get(img)
    if len(faces) == 0:
        return None
    face = get_best_face(faces)
    return face.normed_embedding


def cosine_similarity(emb1, emb2):
    """Similitud coseno entre dos embeddings ya normalizados."""
    return float(np.dot(emb1, emb2))


def guardar_debug_jpg(directorio_fotos, img, motivo):
    """Guarda copia debug de la imagen SOLO en caso de error.
    motivo: texto corto para el nombre del archivo (ej: 'no_rostro', 'no_coincide')
    """
    try:
        debug_dir = os.path.join(directorio_fotos, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_path = os.path.join(debug_dir, f"{motivo}_{ts}.jpg")
        cv2.imwrite(debug_path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        logging.warning(f"DEBUG guardado: {debug_path}")
    except Exception as e:
        logging.warning(f"No se pudo guardar copia debug: {e}")


@app.post("/test")
async def ping():
    application_path = get_application_path()
    logging.info("Test endpoint was pinged.")
    return {"message": f"test ok, application_path = {application_path}", "config": CONFIG}


@app.post("/register")
async def register(request: RegisterRequest):
    """Registra el rostro de un empleado."""
    try:
        directorio_fotos = request.directorio_fotos
        if not os.path.exists(directorio_fotos):
            os.makedirs(directorio_fotos, exist_ok=True)

        img_data = base64.b64decode(request.image)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            logging.error(f"REGISTER {directorio_fotos}: error decodificando imagen")
            raise HTTPException(status_code=400, detail="Error reading image")

        img_norm = normalize_image_size(img)
        faces = face_app.get(img_norm)

        if len(faces) == 0:
            # ERROR: no se detecto rostro -> log + debug
            h, w = img.shape[:2]
            logging.error(f"REGISTER {directorio_fotos}: NO se detecto rostro en imagen {w}x{h}")
            guardar_debug_jpg(directorio_fotos, img, "register_no_rostro")
            raise HTTPException(status_code=400, detail="No se detecto rostro en la imagen")

        face = get_best_face(faces)
        embedding = face.normed_embedding

        cache_path = os.path.join(directorio_fotos, EMBEDDING_FILE)
        np.save(cache_path, embedding)

        folder_name = os.path.basename(os.path.normpath(directorio_fotos))
        jpg_path = os.path.join(directorio_fotos, f"0_{folder_name}.jpg")
        cv2.imwrite(jpg_path, img_norm, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # OK: silencio (no se loguea)
        return {"success": True, "message": "Registro exitoso"}

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error en register: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload(request: UploadRequest):
    """Valida una foto contra el embedding cacheado de un empleado."""
    try:
        directorio_fotos = request.directorio_fotos

        if not os.path.exists(directorio_fotos):
            logging.warning(f"UPLOAD {directorio_fotos}: la carpeta no existe.")
            return [{"name": "No encontrado", "distance": 0.0, "percentage": 0.0}]

        image_base64string = request.image
        img_data = base64.b64decode(image_base64string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            logging.error(f"UPLOAD {directorio_fotos}: error decodificando imagen")
            raise HTTPException(status_code=400, detail="Error reading image")

        cache_path = os.path.join(directorio_fotos, EMBEDDING_FILE)

        # Self-healing: regenerar embedding desde JPG si no existe el .npy
        if not os.path.isfile(cache_path):
            folder_name = os.path.basename(os.path.normpath(directorio_fotos))
            jpg_path = os.path.join(directorio_fotos, f"0_{folder_name}.jpg")

            if os.path.isfile(jpg_path):
                logging.warning(f"UPLOAD {directorio_fotos}: embedding.npy no existe, regenerando desde {jpg_path}")
                try:
                    ref_img = cv2.imread(jpg_path)
                    if ref_img is not None:
                        ref_img = normalize_image_size(ref_img)
                        ref_faces = face_app.get(ref_img)
                        if len(ref_faces) > 0:
                            ref_face = get_best_face(ref_faces)
                            regenerated = ref_face.normed_embedding
                            np.save(cache_path, regenerated)
                            logging.info(f"UPLOAD {directorio_fotos}: embedding regenerado OK")
                        else:
                            logging.error(f"UPLOAD {directorio_fotos}: no se detecto rostro en JPG referencia")
                            return [{"name": "No encontrado", "distance": 0.0, "percentage": 0.0}]
                    else:
                        logging.error(f"UPLOAD {directorio_fotos}: no se pudo leer {jpg_path}")
                        return [{"name": "No encontrado", "distance": 0.0, "percentage": 0.0}]
                except Exception as e:
                    logging.error(f"UPLOAD {directorio_fotos}: error regenerando embedding desde JPG: {e}")
                    return [{"name": "No encontrado", "distance": 0.0, "percentage": 0.0}]
            else:
                logging.error(f"UPLOAD {directorio_fotos}: no hay embedding.npy ni 0_{folder_name}.jpg")
                return [{"name": "No encontrado", "distance": 0.0, "percentage": 0.0}]

        try:
            known_embedding = np.load(cache_path)
        except Exception as e:
            logging.error(f"UPLOAD {directorio_fotos}: error leyendo embedding.npy: {e}")
            return [{"name": "No encontrado", "distance": 0.0, "percentage": 0.0}]

        img_norm = normalize_image_size(img)
        faces = face_app.get(img_norm)

        if len(faces) == 0:
            # ERROR: no se detecto rostro -> log + debug (con la imagen ORIGINAL recibida)
            h, w = img.shape[:2]
            logging.error(f"UPLOAD {directorio_fotos}: NO se detecto rostro en imagen {w}x{h}")
            guardar_debug_jpg(directorio_fotos, img, "no_rostro")
            return [{"name": "No encontrado", "distance": 0.0, "percentage": 0.0}]

        face = get_best_face(faces)
        query_embedding = face.normed_embedding
        sim = cosine_similarity(query_embedding, known_embedding)
        distance = float(1 - sim)
        percentage = float(sim * 100)

        threshold = CONFIG['threshold']
        folder_name = os.path.basename(os.path.normpath(directorio_fotos))
        is_match = sim >= threshold
        name = folder_name if is_match else "No encontrado"

        if not is_match:
            # NO COINCIDE: log + debug
            bbox_w = face.bbox[2] - face.bbox[0]
            bbox_h = face.bbox[3] - face.bbox[1]
            logging.warning(
                f"UPLOAD {directorio_fotos}: NO COINCIDE | "
                f"rostro {bbox_w:.0f}x{bbox_h:.0f} | "
                f"similitud={sim:.4f} | threshold={threshold}"
            )
            guardar_debug_jpg(directorio_fotos, img, "no_coincide")

        # COINCIDE: silencio total (no log, no debug)
        return [{"name": name, "distance": distance, "percentage": percentage}]

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Exception occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    try:
        logging.info("---------------------------------------------------------------")
        logging.info(f"INICIANDO ApiReconocimiento en {CONFIG['host']}:{CONFIG['port']}")
        init_face_engine()
        uvicorn.run(app, host=CONFIG['host'], port=CONFIG['port'], log_config=None)
    except Exception as e:
        print(f"Error : {str(e)}")
        logging.info(f"Error : {str(e)}")