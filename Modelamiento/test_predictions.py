#!/usr/bin/env python3
"""
Script para probar las predicciones con los modelos entrenados.
Carga los modelos guardados y hace predicciones de ejemplo.

Uso:
    python test_predictions.py                    # Prueba con todas las tiendas entrenadas
    python test_predictions.py --store COBU3_11475795  # Prueba con una tienda específica
    python test_predictions.py --samples 10       # Muestra 10 predicciones por tienda
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import joblib
import warnings
warnings.filterwarnings('ignore')

# Agregar el directorio model al path
model_dir = Path(__file__).parent / "model"
sys.path.append(str(model_dir))

from processing.data_manager import load_dataset
from processing.features_improved import get_improved_feature_columns
from pipeline_improved import lifemiles_improved_preprocessing_pipe
from config.core import config


def load_trained_models():
    """Cargar todos los modelos entrenados disponibles."""
    models_dir = Path(__file__).parent / "model" / "trained" / "store_models"
    models = {}
    
    print("Cargando modelos entrenados...")
    
    for model_file in models_dir.glob("model_*.pkl"):
        # Extraer partner_code y store_code del nombre del archivo
        # Formato: model_PARTNER_STORE_VERSION.pkl
        parts = model_file.stem.split("_")
        if len(parts) >= 3:
            partner_code = parts[1]
            store_code = parts[2]
            store_key = f"{partner_code}_{store_code}"
            
            try:
                model = joblib.load(model_file)
                models[store_key] = {
                    'model': model,
                    'partner_code': partner_code,
                    'store_code': store_code,
                    'file_path': model_file
                }
                print(f"  ✓ Modelo cargado: {store_key}")
            except Exception as e:
                print(f"  ✗ Error cargando {model_file}: {e}")
    
    print(f"\nTotal modelos cargados: {len(models)}")
    return models


def prepare_sample_data(data, store_key, partner_code, store_code, n_samples=5):
    """Preparar datos de muestra para predicción."""
    
    # Filtrar datos para la tienda específica
    store_data = data[
        (data['partner_code'] == partner_code) & 
        (data['store_code'] == store_code)
    ].copy()
    
    if len(store_data) == 0:
        print(f"  ⚠️  No hay datos disponibles para {store_key}")
        return None
    
    # Tomar las últimas muestras disponibles
    store_data = store_data.sort_values('date').tail(n_samples)
    
    print(f"  📊 Datos de muestra para {store_key}: {len(store_data)} registros")
    print(f"     Período: {store_data['date'].min()} a {store_data['date'].max()}")
    
    return store_data


def make_predictions(model_info, sample_data):
    """Hacer predicciones con un modelo específico."""
    
    try:
        # Aplicar preprocesamiento
        processed_data = lifemiles_improved_preprocessing_pipe.transform(sample_data)
        
        # Obtener columnas de características
        feature_columns = get_improved_feature_columns()
        exclude_columns = ['partner_code', 'partner_name', 'store_code', 'store_name', 'date', config.model_config.target]
        numeric_features = [col for col in feature_columns if col not in exclude_columns]
        
        # Seleccionar solo las características numéricas disponibles
        available_features = [col for col in numeric_features if col in processed_data.columns]
        X = processed_data[available_features]
        
        # Hacer predicciones
        model = model_info['model']
        predictions = model.predict(X)
        
        # Preparar resultados
        results = pd.DataFrame({
            'date': sample_data['date'].values,
            'actual': sample_data[config.model_config.target].values,
            'predicted': predictions,
            'error': sample_data[config.model_config.target].values - predictions,
            'error_pct': ((sample_data[config.model_config.target].values - predictions) / 
                         sample_data[config.model_config.target].values * 100)
        })
        
        return results
        
    except Exception as e:
        print(f"  ✗ Error en predicción: {e}")
        return None


def display_prediction_results(store_key, results):
    """Mostrar resultados de predicción de forma clara."""
    
    print(f"\n{'='*60}")
    print(f"PREDICCIONES PARA TIENDA: {store_key}")
    print(f"{'='*60}")
    
    if results is None or len(results) == 0:
        print("No hay resultados para mostrar")
        return
    
    # Estadísticas de resumen
    mae = np.mean(np.abs(results['error']))
    rmse = np.sqrt(np.mean(results['error']**2))
    mape = np.mean(np.abs(results['error_pct']))
    
    print(f"📈 MÉTRICAS DE EVALUACIÓN:")
    print(f"   MAE (Error Absoluto Medio): {mae:.2f}")
    print(f"   RMSE (Raíz Error Cuadrático): {rmse:.2f}")
    print(f"   MAPE (Error Porcentual Absoluto): {mape:.2f}%")
    
    print(f"\n📋 PREDICCIONES DETALLADAS:")
    print("-" * 80)
    print(f"{'Fecha':<12} {'Real':<10} {'Predicho':<10} {'Error':<10} {'Error %':<10}")
    print("-" * 80)
    
    for _, row in results.iterrows():
        print(f"{str(row['date']):<12} {row['actual']:<10.2f} {row['predicted']:<10.2f} "
              f"{row['error']:<10.2f} {row['error_pct']:<10.2f}%")
    
    print("-" * 80)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test predictions with trained LifeMiles models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_predictions.py                           # Test all trained stores
  python test_predictions.py --store COBU3_11475795   # Test specific store
  python test_predictions.py --samples 10             # Show 10 predictions per store
        """
    )
    
    parser.add_argument(
        '--store',
        type=str,
        default=None,
        help='Specific store to test (format: PARTNER_STORE)'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=5,
        help='Number of sample predictions to show per store (default: 5)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def main():
    """Execute prediction testing."""
    
    args = parse_arguments()
    
    print("LIFEMILES PREDICTION TESTING")
    print("="*50)
    print("Testing trained models with sample predictions")
    print("="*50)
    
    try:
        # Cargar modelos entrenados
        models = load_trained_models()
        
        if not models:
            print("❌ No hay modelos entrenados disponibles")
            return
        
        # Cargar datos
        print("\nCargando datos...")
        data = load_dataset(file_name=config.app_config.train_data_file)
        print(f"Datos cargados: {data.shape}")
        
        # Filtrar tienda específica si se especifica
        stores_to_test = [args.store] if args.store else list(models.keys())
        
        if args.store and args.store not in models:
            print(f"❌ Modelo para tienda {args.store} no encontrado")
            print(f"Tiendas disponibles: {list(models.keys())}")
            return
        
        print(f"\nProbando {len(stores_to_test)} tienda(s)...")
        
        # Probar cada tienda
        for store_key in stores_to_test:
            if store_key not in models:
                print(f"⚠️  Modelo para {store_key} no disponible")
                continue
                
            model_info = models[store_key]
            
            # Preparar datos de muestra
            sample_data = prepare_sample_data(
                data, 
                store_key, 
                model_info['partner_code'], 
                model_info['store_code'],
                args.samples
            )
            
            if sample_data is None:
                continue
            
            # Hacer predicciones
            results = make_predictions(model_info, sample_data)
            
            # Mostrar resultados
            display_prediction_results(store_key, results)
        
        print(f"\n✅ TESTING COMPLETADO")
        print(f"Modelos probados: {len([s for s in stores_to_test if s in models])}")
        
    except Exception as e:
        print(f"❌ Error durante el testing: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
