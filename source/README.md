# ApiReconocimiento (InsightFace)

API de reconocimiento facial local usando InsightFace + ONNX Runtime.
Mantiene los mismos endpoints y formato JSON que la version anterior basada en face_recognition/dlib.

## Endpoints

- `POST /test` - ping de prueba
- `POST /upload` - recibe `{ "directorio_fotos": "...", "image": "base64..." }` y devuelve `[{ "name", "distance", "percentage" }]`

## Instalacion

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

Nada de CMake, Visual Studio, ni dlib. Solo esto.

### 2. Primera ejecucion (descarga de modelos)

La primera vez que corras el script, InsightFace descarga los modelos automaticamente (~300 MB) a:

```
Windows: C:\Users\TU_USUARIO\.insightface\models\buffalo_l\
```

Necesita internet solo la primera vez. Despues corre 100% local.

probado en Python 3.13.13

```bash
python ApiReconocimiento.py
```

Veras en la consola algo como "download_path: ...buffalo_l". Espera a que termine.

### 3. Configuracion

El puerto se define en `config.ini`:

```ini
[server]
port = 5000
```

## Empaquetar como .exe

### Paso 1: asegurarse de que los modelos estan descargados

Corre `python ApiReconocimiento.py` al menos una vez para que descargue los modelos en tu carpeta de usuario.

### Paso 2: construir

```bash
pip install pyinstaller
pyinstaller ApiReconocimiento.spec

###nuitka installer
python -m nuitka ApiReconocimiento.py --standalone --include-package=cv2 --include-package=insightface --include-package=onnxruntime --include-package=fastapi --include-package=uvicorn --include-package=starlette --include-package=pydantic --include-data-files=config.ini=config.ini --output-dir=dist_nuitka --windows-console-mode=attach

python -m nuitka ApiReconocimiento.py --standalone --enable-plugin=numpy --enable-plugin=cv2 --include-package=cv2 --include-package=insightface --include-package=onnxruntime --include-package=fastapi --include-package=uvicorn --include-package=starlette --include-package=pydantic --include-module=PIL._imaging --include-module=PIL.JpegImagePlugin --include-module=PIL.PngImagePlugin --include-module=PIL.Image --include-data-files=config.ini=config.ini --include-data-dir=models=models --output-dir=dist_nuitka --windows-console-mode=attach

python -m nuitka --standalone --enable-plugin=numpy --enable-plugin=cv2 --include-package=insightface --include-package=onnxruntime --include-data-dir=models=./models

El .exe queda en `dist/ApiReconocimiento.exe`. Copia tambien `config.ini` a esa carpeta.

El spec file ya incluye los modelos dentro del .exe, asi que el ejecutable final funciona sin internet.

## Umbral de coincidencia

Dentro de `ApiReconocimiento.py` hay una constante:

```python
THRESHOLD = 0.40
```

Es la similitud coseno minima (0 a 1) para considerar que hay match.
- Mas alto (ej: 0.50) = mas estricto, menos falsos positivos.
- Mas bajo (ej: 0.30) = mas permisivo, mas falsos positivos.

Haz pruebas con tus fotos reales y ajusta.

## Formato de respuesta

Igual que antes:

```json
[
  { "name": "juan_perez", "distance": 0.35, "percentage": 65.0 }
]
```

- `name`: nombre del archivo (sin extension) de la foto que hizo match, o "No encontrado".
- `distance`: 1 - similitud_coseno (menor = mas parecido).
- `percentage`: similitud_coseno * 100.

**Nota:** los valores numericos NO son iguales a los de face_recognition anterior. Si tu cliente tenia un umbral fijo (ej: percentage > 60), prueba y recalibra.

