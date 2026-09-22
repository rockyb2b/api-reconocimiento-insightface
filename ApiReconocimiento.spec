# -*- mode: python ; coding: utf-8 -*-
# IMPORTANTE: antes de construir el .exe, corre el script una vez con
# "python ApiReconocimiento.py" para que InsightFace descargue los modelos
# en ~/.insightface/models/buffalo_l/ (Windows: C:\Users\TU_USUARIO\.insightface\models\buffalo_l\)
# Esos modelos se copian dentro del .exe con el bloque "datas" de abajo.


import os, shutil
import cv2
from pathlib import Path
import onnxruntime
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_data_files
import onnxruntime as ort
base = os.path.abspath(os.getcwd())
from PyInstaller.utils.hooks import collect_dynamic_libs
# ONNX paths
#onnx_path = os.path.dirname(onnxruntime.__file__)
#onnx_capi_path = os.path.join(onnx_path, 'capi')

# InsightFace models
home = str(Path.home())
insightface_models = os.path.join(home, '.insightface', 'models', 'buffalo_l')

datas = []

datas += collect_data_files('cv2')
datas += collect_data_files('insightface')#########
# 🔥 FORCE include cv2 data folder (IMPORTANT)
cv2_path = os.path.dirname(cv2.__file__)
datas.append((os.path.join(cv2_path, 'data'), 'cv2/data'))

binaries = []
binaries += collect_dynamic_libs('onnxruntime')
binaries += collect_dynamic_libs('cv2')
binaries += collect_dynamic_libs('numpy')

binaries += collect_dynamic_libs('onnxruntime')

# FORCE providers DLL stability
for p in ort.get_available_providers():
    print("Provider:", p)

# Include InsightFace models
if os.path.exists(insightface_models):
    datas.append((insightface_models, 'insightface/models/buffalo_l'))

# Include ONNXRuntime DLLs (ONLY DLLs, not whole folder)
#binaries.append((os.path.join(onnx_capi_path, '*.dll'), 'onnxruntime/capi'))


cv2_hidden = collect_submodules('cv2')

opencv_hidden = [
    'cv2',
    'cv2.data',
    'cv2.gapi',
    'cv2.typing',
    'cv2.utils',
    'cv2.config',
    'cv2.mat_wrapper',
    'cv2.misc',
]

a = Analysis(
    ['ApiReconocimiento.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
         'fastapi',
        'starlette',
        'pydantic',
        'uvicorn',
        'uvicorn.logging',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',
        'insightface',
        'insightface.app',
        'insightface.model_zoo',
        'onnxruntime',
        'numpy',
        'cv2.data',
        
    ] + cv2_hidden + opencv_hidden,   # 👈 HERE,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,   # 👈 IMPORTANT
    name='ApiReconocimientoInsightFace',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='ApiReconocimientoInsightFace',
)


# ✅ Manual copy of data files to the exe directory
def copy_data_files():
    dist_dir = os.path.join(base, 'dist', 'ApiReconocimientoInsightFace')
    files_to_copy = [
        'config.ini',
        'msvcp140.dll'
    ]
    
    for file in files_to_copy:
        src = os.path.join(base, file)
        if os.path.exists(src):
            shutil.copy2(src, dist_dir)
            print(f"Copied {file} to {dist_dir}")

# Call the function after build
copy_data_files()   