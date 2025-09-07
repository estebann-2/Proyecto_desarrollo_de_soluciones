#!/usr/bin/env python3
"""
Script de diagnóstico para verificar las rutas utilizadas en el proyecto.
"""

import sys
import os
from pathlib import Path

# Agregar el directorio model al path
model_dir = Path(__file__).parent / "model"
sys.path.append(str(model_dir))

print("=== DIAGNÓSTICO DE RUTAS ===")
print(f"Script ejecutándose desde: {Path(__file__).parent}")
print(f"Working directory: {os.getcwd()}")
print(f"Python executable: {sys.executable}")
print()

try:
    # Importar el módulo model
    import model
    print(f"model.__file__: {model.__file__}")
    
    # Verificar configuración de rutas
    from config.core import PACKAGE_ROOT, ROOT, CONFIG_FILE_PATH, DATASET_DIR, TRAINED_MODEL_DIR
    
    print("\n=== RUTAS CONFIGURADAS ===")
    print(f"PACKAGE_ROOT: {PACKAGE_ROOT}")
    print(f"ROOT: {ROOT}")
    print(f"CONFIG_FILE_PATH: {CONFIG_FILE_PATH}")
    print(f"DATASET_DIR: {DATASET_DIR}")
    print(f"TRAINED_MODEL_DIR: {TRAINED_MODEL_DIR}")
    
    print("\n=== VERIFICACIÓN DE EXISTENCIA ===")
    print(f"PACKAGE_ROOT exists: {PACKAGE_ROOT.exists()}")
    print(f"ROOT exists: {ROOT.exists()}")
    print(f"CONFIG_FILE_PATH exists: {CONFIG_FILE_PATH.exists()}")
    print(f"DATASET_DIR exists: {DATASET_DIR.exists()}")
    print(f"TRAINED_MODEL_DIR exists: {TRAINED_MODEL_DIR.exists()}")
    
    print("\n=== VERIFICACIÓN DE PERMISOS ===")
    print(f"PACKAGE_ROOT writable: {os.access(PACKAGE_ROOT, os.W_OK)}")
    print(f"ROOT writable: {os.access(ROOT, os.W_OK)}")
    print(f"TRAINED_MODEL_DIR writable: {os.access(TRAINED_MODEL_DIR, os.W_OK)}")
    
    # Verificar permisos de la carpeta padre
    print(f"Parent of TRAINED_MODEL_DIR writable: {os.access(TRAINED_MODEL_DIR.parent, os.W_OK)}")
    
    # Intentar crear directorio de prueba
    test_dir = TRAINED_MODEL_DIR / "test_permissions"
    try:
        test_dir.mkdir(exist_ok=True)
        test_file = test_dir / "test.txt"
        test_file.write_text("test")
        print(f"✅ Test write successful in: {test_dir}")
        
        # Limpiar
        test_file.unlink()
        test_dir.rmdir()
        
    except Exception as e:
        print(f"❌ Test write failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Attempted path: {test_dir}")
    
    # Verificar estructura de directorios de modelo
    print("\n=== ESTRUCTURA DE DIRECTORIOS ===")
    store_models_dir = TRAINED_MODEL_DIR / "store_models"
    print(f"store_models directory: {store_models_dir}")
    print(f"store_models exists: {store_models_dir.exists()}")
    if store_models_dir.exists():
        print(f"store_models writable: {os.access(store_models_dir, os.W_OK)}")

except Exception as e:
    print(f"❌ Error en importación: {e}")
    print(f"   Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()

print("\n=== VARIABLES DE ENTORNO RELEVANTES ===")
for var in ['HOME', 'USER', 'PWD', 'PYTHONPATH']:
    value = os.environ.get(var, 'NOT SET')
    print(f"{var}: {value}")

print("\n=== FIN DIAGNÓSTICO ===")
