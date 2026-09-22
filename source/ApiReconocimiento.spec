# -*- mode: python ; coding: utf-8 -*-
# ApiReconocimiento.spec
#
# Para construir: pyinstaller ApiReconocimiento.spec
#
# IMPORTANTE: antes de construir el .exe, corre el script una vez con
# "python ApiReconocimiento.py" para que InsightFace descargue los modelos
# en ~/.insightface/models/buffalo_l/ (Windows: C:\Users\TU_USUARIO\.insightface\models\buffalo_l\)
# Esos modelos se copian dentro del .exe con el bloque "datas" de abajo.

import os
from pathlib import Path

# Ruta a los modelos descargados por InsightFace
home = str(Path.home())
insightface_models = os.path.join(home, '.insightface', 'models', 'buffalo_l')

datas = []
if os.path.exists(insightface_models):
    datas.append((insightface_models, 'insightface/models/buffalo_l'))

a = Analysis(
    ['ApiReconocimiento.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'insightface',
        'onnxruntime',
        'cv2',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ApiReconocimiento',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
