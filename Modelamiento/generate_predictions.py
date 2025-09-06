#!/usr/bin/env python3
"""
Script para generar predicciones futuras con los modelos entrenados.
Crea predicciones para fechas nuevas (futuras) que no están en los datos históricos.

Uso:
    python generate_predictions.py                        # Predicciones para todas las tiendas, próximos 7 días
    python generate_predictions.py --days 30             # Predicciones para próximos 30 días
    python generate_predictions.py --store COBU3_11475795 # Predicciones para tienda específica
    python generate_predictions.py --start-date 2024-01-01 # Desde fecha específica
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
                print(f"  Modelo cargado: {store_key}")
            except Exception as e:
                print(f"  Error cargando {model_file}: {e}")
    
    print(f"Total modelos cargados: {len(models)}")
    return models


def generate_future_dates(start_date, days):
    """Generar fechas futuras para predicción."""
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    
    dates = []
    for i in range(days):
        future_date = start_date + timedelta(days=i)
        dates.append(future_date.strftime('%Y-%m-%d'))
    
    return dates


def create_future_data_template(historical_data, store_key, partner_code, store_code, future_dates):
    """Crear plantilla de datos futuros basada en datos históricos."""
    
    # Filtrar datos históricos de la tienda
    store_data = historical_data[
        (historical_data['partner_code'] == partner_code) & 
        (historical_data['store_code'] == store_code)
    ].copy()
    
    if len(store_data) == 0:
        print(f"  No hay datos disponibles para {store_key}")
        return None
    
    # Analizar patrones históricos
    latest_data = store_data.sort_values('date').tail(60)  # Últimos 60 días como referencia
    
    # Calcular estadísticas por día de semana
    store_data['weekday'] = pd.to_datetime(store_data['date']).dt.dayofweek
    target_col = config.model_config.target  # 'billings'
    weekday_stats = store_data.groupby('weekday')[target_col].agg(['mean', 'std']).fillna(0)
    
    # Estadísticas generales
    overall_mean = store_data[target_col].mean()
    overall_std = store_data[target_col].std()
    
    # Crear datos futuros
    future_data = []
    
    for future_date in future_dates:
        # Crear una fila base con características estructurales de la tienda
        template_row = latest_data.iloc[-1].copy()
        
        # Actualizar fecha y características temporales
        date_obj = datetime.strptime(future_date, '%Y-%m-%d')
        template_row['date'] = date_obj
        
        # Características temporales que deben cambiar
        weekday = date_obj.weekday()
        
        # Simular billings basado en patrones históricos
        if weekday in weekday_stats.index:
            day_mean = weekday_stats.loc[weekday, 'mean']
            day_std = weekday_stats.loc[weekday, 'std']
        else:
            day_mean = overall_mean
            day_std = overall_std
        
        # Generar billings variable basado en día de semana + ruido
        np.random.seed(hash(future_date + store_key) % (2**31))  # Seed determinista pero variable
        simulated_billings = max(0, np.random.normal(day_mean, day_std * 0.3))
        
        template_row[target_col] = simulated_billings
        
        # Actualizar características que dependen de la fecha
        template_row['year'] = date_obj.year
        template_row['month'] = date_obj.month
        template_row['day'] = date_obj.day
        template_row['quarter'] = (date_obj.month - 1) // 3 + 1
        template_row['week_of_year'] = date_obj.isocalendar()[1]
        template_row['day_of_week'] = weekday
        template_row['day_of_month'] = date_obj.day
        template_row['is_weekend'] = 1 if weekday >= 5 else 0
        template_row['is_month_start'] = 1 if date_obj.day <= 3 else 0
        template_row['is_month_end'] = 1 if date_obj.day >= 28 else 0
        
        # Simular algunas otras características variables
        if 'promocion_activa' in template_row:
            # Simular promociones de forma aleatoria pero consistente
            promo_seed = hash(f"promo_{future_date}_{store_key}") % 100
            template_row['promocion_activa'] = 1 if promo_seed < 20 else 0  # 20% probabilidad
        
        # Agregar algo de variabilidad a características numéricas existentes
        numeric_cols = ['precio_promedio', 'descuento_aplicado', 'num_transacciones']
        for col in numeric_cols:
            if col in template_row and pd.notna(template_row[col]):
                noise_factor = np.random.uniform(0.95, 1.05)
                template_row[col] = template_row[col] * noise_factor
        
        future_data.append(template_row)
    
    future_df = pd.DataFrame(future_data)
    future_df['date'] = pd.to_datetime(future_df['date'])
    
    print(f"  Datos futuros creados para {store_key}: {len(future_df)} fechas")
    print(f"     Período: {future_df['date'].min().date()} a {future_df['date'].max().date()}")
    print(f"     Rango {config.model_config.target}: ${future_df[config.model_config.target].min():.2f} - ${future_df[config.model_config.target].max():.2f}")
    
    return future_df


def generate_predictions_for_store(model_info, future_data):
    """Generar predicciones para una tienda específica."""
    
    store_key = f"{model_info['partner_code']}_{model_info['store_code']}"
    
    try:
        print(f"  Generando predicciones para {store_key}...")
        
        # Aplicar preprocesamiento (sin target ya que son datos futuros)
        processed_data = lifemiles_improved_preprocessing_pipe.transform(future_data)
        
        # DEBUG: Mostrar algunas características
        print(f"    DEBUG - Características antes del modelo:")
        print(f"      Fechas únicas: {future_data['date'].nunique()}")
        print(f"      {config.model_config.target} rango: {future_data[config.model_config.target].min():.2f} - {future_data[config.model_config.target].max():.2f}")
        
        # Obtener columnas de características
        feature_columns = get_improved_feature_columns()
        exclude_columns = ['partner_code', 'partner_name', 'store_code', 'store_name', 'date', config.model_config.target]
        numeric_features = [col for col in feature_columns if col not in exclude_columns]
        
        # Seleccionar solo las características numéricas disponibles
        available_features = [col for col in numeric_features if col in processed_data.columns]
        X = processed_data[available_features]
        
        print(f"      Características usadas: {len(available_features)}")
        print(f"      Primeras 3 características: {available_features[:3]}")
        
        # Verificar variabilidad en las características
        if len(X) > 1:
            variability = X.std().sum()
            print(f"      Variabilidad total (suma std): {variability:.4f}")
        
        # Hacer predicciones
        model = model_info['model']
        predictions = model.predict(X)
        
        print(f"      Predicciones rango: {predictions.min():.4f} - {predictions.max():.4f}")
        
        # Preparar resultados
        results = pd.DataFrame({
            'date': future_data['date'].dt.strftime('%Y-%m-%d'),
            'partner_code': model_info['partner_code'],
            'store_code': model_info['store_code'],
            'predicted_monto': predictions,
            'day_of_week': future_data['date'].dt.day_name(),
            'month': future_data['date'].dt.month,
            'quarter': future_data['date'].dt.quarter
        })
        
        return results
        
    except Exception as e:
        print(f"  Error generando predicciones para {store_key}: {e}")
        return None


def save_predictions(all_predictions, output_file="predicciones_futuras.csv"):
    """Guardar predicciones en archivo CSV."""
    
    if not all_predictions:
        print("No hay predicciones para guardar")
        return
    
    # Combinar todas las predicciones
    combined_df = pd.concat(all_predictions, ignore_index=True)
    
    # Guardar archivo
    output_path = Path(__file__).parent / output_file
    combined_df.to_csv(output_path, index=False)
    
    print(f"\nPredicciones guardadas en: {output_path}")
    print(f"   Total registros: {len(combined_df)}")
    print(f"   Tiendas: {combined_df['store_code'].nunique()}")
    print(f"   Período: {combined_df['date'].min()} a {combined_df['date'].max()}")


def display_predictions_summary(all_predictions):
    """Mostrar predicciones en formato de tabla simple."""
    
    if not all_predictions:
        print("No hay predicciones para mostrar")
        return
    
    combined_df = pd.concat(all_predictions, ignore_index=True)
    
    print(f"\n{'='*50}")
    print(f"PREDICCIONES FUTURAS")
    print(f"{'='*50}")
    
    # Mostrar tabla simple para cada tienda
    for store_code in combined_df['store_code'].unique():
        store_preds = combined_df[combined_df['store_code'] == store_code]
        partner_code = store_preds['partner_code'].iloc[0]
        
        print(f"\nTIENDA: {partner_code}_{store_code}")
        print("-" * 40)
        print(f"{'Fecha':<12} {'Predicción'}")
        print("-" * 40)
        
        # Mostrar todas las predicciones
        for _, row in store_preds.iterrows():
            print(f"{row['date']:<12} ${row['predicted_monto']:>10,.2f}")
        
        print("-" * 40)
        print(f"Total: ${store_preds['predicted_monto'].sum():>10,.2f}")
        print("")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate future predictions with trained LifeMiles models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_predictions.py                           # 7 days for all stores
  python generate_predictions.py --days 30                # 30 days for all stores
  python generate_predictions.py --store COBU3_11475795   # Specific store
  python generate_predictions.py --start-date 2024-01-01  # From specific date
        """
    )
    
    parser.add_argument(
        '--store',
        type=str,
        default=None,
        help='Specific store to predict (format: PARTNER_STORE)'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Number of days to predict (default: 7)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Start date for predictions (YYYY-MM-DD). Default: tomorrow'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default="predicciones_futuras.csv",
        help='Output file name (default: predicciones_futuras.csv)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def main():
    """Execute future predictions generation."""
    
    args = parse_arguments()
    
    print("LIFEMILES FUTURE PREDICTIONS GENERATOR")
    print("="*50)
    print("Generating new predictions for future dates")
    print("="*50)
    
    try:
        # Determinar fecha de inicio
        if args.start_date:
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        else:
            start_date = datetime.now() + timedelta(days=1)  # Mañana
        
        print(f"Período de predicción: {start_date.date()} por {args.days} días")
        
        # Cargar modelos entrenados
        models = load_trained_models()
        
        if not models:
            print("No hay modelos entrenados disponibles")
            return
        
        # Cargar datos históricos
        print("\nCargando datos históricos...")
        historical_data = load_dataset(file_name=config.app_config.train_data_file)
        print(f"Datos históricos: {historical_data.shape}")
        
        # Generar fechas futuras
        future_dates = generate_future_dates(start_date, args.days)
        print(f"Fechas a predecir: {len(future_dates)}")
        
        # Filtrar tiendas si se especifica
        stores_to_predict = [args.store] if args.store else list(models.keys())
        
        if args.store and args.store not in models:
            print(f"Modelo para tienda {args.store} no encontrado")
            print(f"Tiendas disponibles: {list(models.keys())}")
            return
        
        print(f"\nGenerando predicciones para {len(stores_to_predict)} tienda(s)...")
        
        all_predictions = []
        
        # Generar predicciones para cada tienda
        for store_key in stores_to_predict:
            if store_key not in models:
                print(f"Modelo para {store_key} no disponible")
                continue
                
            model_info = models[store_key]
            
            # Crear datos futuros
            future_data = create_future_data_template(
                historical_data,
                store_key,
                model_info['partner_code'],
                model_info['store_code'],
                future_dates
            )
            
            if future_data is None:
                continue
            
            # Generar predicciones
            predictions = generate_predictions_for_store(model_info, future_data)
            
            if predictions is not None:
                all_predictions.append(predictions)
        
        # Mostrar resumen
        display_predictions_summary(all_predictions)
        
        # Guardar predicciones
        save_predictions(all_predictions, args.output)
        
        print(f"\nPREDICCIONES FUTURAS GENERADAS EXITOSAMENTE")
        print(f"Tiendas procesadas: {len(all_predictions)}")
        print(f"Total predicciones: {sum(len(pred) for pred in all_predictions)}")
        
    except Exception as e:
        print(f"Error durante la generación: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
