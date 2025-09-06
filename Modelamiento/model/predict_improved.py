"""
Módulo de predicción mejorado para el modelo LifeMiles.
Incorpora todas las mejoras del pipeline: transformación de target,
características avanzadas y manejo robusto de errores.
"""

import typing as t
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from model import __version__ as _version
from model.config.core import config
from model.processing.data_manager import (
    load_pipeline, 
    load_store_model, 
    get_available_stores
)
from model.processing.validation import validate_inputs
from model.processing.features_improved import get_improved_feature_columns, TargetTransformer

# Cargar pipeline mejorado de preprocesamiento
preprocessing_pipeline_file_name = f"{config.app_config.pipeline_save_file}_improved_preprocessing_{_version}.pkl"
try:
    _improved_preprocessing_pipe = load_pipeline(file_name=preprocessing_pipeline_file_name)
except:
    # Fallback al pipeline original si no existe el mejorado
    preprocessing_pipeline_file_name = f"{config.app_config.pipeline_save_file}_preprocessing_{_version}.pkl"
    _improved_preprocessing_pipe = load_pipeline(file_name=preprocessing_pipeline_file_name)


def create_forecast_features_improved(
    historical_data: pd.DataFrame,
    forecast_dates: pd.DatetimeIndex,
    partner_code: str,
    store_code: str
) -> pd.DataFrame:
    """Create improved features for forecasting future dates."""
    
    # Crear DataFrame con las fechas de forecast
    forecast_df = pd.DataFrame({
        config.model_config.date_column: forecast_dates,
        config.model_config.partner_id_column: partner_code,
        config.model_config.store_id_column: store_code,
        config.model_config.target: np.nan  # Será predicho
    })
    
    # Combinar datos históricos con fechas de forecast
    combined_data = pd.concat([historical_data, forecast_df], ignore_index=True)
    combined_data = combined_data.sort_values(config.model_config.date_column).reset_index(drop=True)
    
    # Aplicar pipeline mejorado de preprocesamiento para crear features
    processed_data = _improved_preprocessing_pipe.transform(combined_data)
    
    # Retornar solo las filas correspondientes al forecast
    forecast_start_idx = len(historical_data)
    forecast_features = processed_data.iloc[forecast_start_idx:].copy()
    
    return forecast_features


def make_prediction_improved(
    *,
    input_data: t.Union[pd.DataFrame, dict],
    partner_code: str,
    store_code: str,
    forecast_days: int = None
) -> dict:
    """
    Make improved predictions with enhanced error handling and metrics.
    
    Args:
        input_data: Historical data for the store
        partner_code: Partner identifier
        store_code: Store identifier  
        forecast_days: Number of days to forecast (default from config)
        
    Returns:
        Dictionary with predictions, confidence intervals, and metrics
    """
    
    if forecast_days is None:
        forecast_days = config.model_config.forecast_days
    
    try:
        # Validar inputs
        if isinstance(input_data, dict):
            input_data = pd.DataFrame(input_data)
        
        validated_data = validate_inputs(input_data=input_data)
        
        # Verificar si el modelo existe para esta tienda
        store_models = get_available_stores()
        store_key = f"{partner_code}_{store_code}"
        
        if store_key not in [f"{s['partner_code']}_{s['store_code']}" for s in store_models]:
            return {
                'predictions': None,
                'errors': f"No trained model available for store {store_code}, partner {partner_code}",
                'version': _version,
                'status': 'model_not_found'
            }
        
        # Cargar modelo específico de la tienda
        model_info = load_store_model(partner_code=partner_code, store_code=store_code)
        
        if model_info is None:
            return {
                'predictions': None,
                'errors': f"Failed to load model for store {store_code}, partner {partner_code}",
                'version': _version,
                'status': 'model_load_error'
            }
        
        model = model_info['model']
        scaler = model_info.get('scaler')
        feature_columns = model_info.get('feature_columns', get_improved_feature_columns())
        
        # Preparar datos históricos para la tienda específica
        store_data = validated_data[
            (validated_data[config.model_config.partner_id_column] == partner_code) &
            (validated_data[config.model_config.store_id_column] == store_code)
        ].copy()
        
        if len(store_data) == 0:
            return {
                'predictions': None,
                'errors': f"No historical data found for store {store_code}, partner {partner_code}",
                'version': _version,
                'status': 'no_data'
            }
        
        # Generar fechas de forecast
        last_date = pd.to_datetime(store_data[config.model_config.date_column]).max()
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )
        
        # Crear features para forecast
        forecast_features = create_forecast_features_improved(
            historical_data=store_data,
            forecast_dates=forecast_dates,
            partner_code=partner_code,
            store_code=store_code
        )
        
        # Filtrar features disponibles
        available_features = [col for col in feature_columns if col in forecast_features.columns]
        
        if len(available_features) == 0:
            return {
                'predictions': None,
                'errors': "No features available for prediction",
                'version': _version,
                'status': 'no_features'
            }
        
        # Preparar datos para predicción
        X_forecast = forecast_features[available_features].fillna(0)
        
        # Aplicar escalado si existe
        if scaler is not None:
            X_forecast_scaled = scaler.transform(X_forecast)
        else:
            X_forecast_scaled = X_forecast
        
        # Realizar predicciones
        predictions_transformed = model.predict(X_forecast_scaled)
        
        # Aplicar transformación inversa si se usó transformación de target
        if hasattr(config.model_config, 'use_target_transformation') and config.model_config.use_target_transformation:
            target_transformer = TargetTransformer()
            predictions = target_transformer.inverse_transform(predictions_transformed)
        else:
            predictions = predictions_transformed
        
        # Asegurar que las predicciones no sean negativas
        predictions = np.maximum(predictions, 0)
        
        # Calcular estadísticas de confianza basadas en datos históricos
        historical_target = store_data[config.model_config.target]
        mean_historical = historical_target.mean()
        std_historical = historical_target.std()
        
        # Intervalos de confianza aproximados (±1.96 * std para 95%)
        confidence_lower = predictions - 1.96 * std_historical
        confidence_upper = predictions + 1.96 * std_historical
        confidence_lower = np.maximum(confidence_lower, 0)  # No negativos
        
        # Crear DataFrame de resultados
        results_df = pd.DataFrame({
            'date': forecast_dates,
            'predicted_billings': predictions,
            'confidence_lower': confidence_lower,
            'confidence_upper': confidence_upper,
            'partner_code': partner_code,
            'store_code': store_code
        })
        
        # Métricas adicionales
        total_forecast = np.sum(predictions)
        avg_daily_forecast = np.mean(predictions)
        
        # Comparación con promedio histórico
        avg_historical = historical_target.mean()
        trend_vs_historical = (avg_daily_forecast - avg_historical) / avg_historical * 100 if avg_historical > 0 else 0
        
        return {
            'predictions': results_df.to_dict('records'),
            'summary': {
                'total_forecast': float(total_forecast),
                'avg_daily_forecast': float(avg_daily_forecast),
                'avg_historical': float(avg_historical),
                'trend_vs_historical_pct': float(trend_vs_historical),
                'forecast_period_days': forecast_days,
                'confidence_level': 95.0
            },
            'model_info': {
                'features_used': len(available_features),
                'total_features_available': len(feature_columns),
                'model_quality': model_info.get('quality', 'Unknown'),
                'test_r2': model_info.get('test_metrics', {}).get('r2', None),
                'test_smape': model_info.get('test_metrics', {}).get('smape', None)
            },
            'version': _version,
            'status': 'success',
            'errors': None
        }
        
    except Exception as e:
        return {
            'predictions': None,
            'errors': str(e),
            'version': _version,
            'status': 'prediction_error'
        }


def make_bulk_predictions_improved(
    *,
    input_data: pd.DataFrame,
    forecast_days: int = None,
    store_list: t.List[t.Tuple[str, str]] = None
) -> dict:
    """
    Make improved predictions for multiple stores.
    
    Args:
        input_data: Historical data for all stores
        forecast_days: Number of days to forecast
        store_list: List of (partner_code, store_code) tuples to predict for
        
    Returns:
        Dictionary with predictions for all stores
    """
    
    if forecast_days is None:
        forecast_days = config.model_config.forecast_days
    
    # Si no se especifica lista de tiendas, usar todas las disponibles
    if store_list is None:
        available_stores = get_available_stores()
        store_list = [(s['partner_code'], s['store_code']) for s in available_stores]
    
    results = {}
    successful_predictions = 0
    
    print(f"🔮 Making predictions for {len(store_list)} stores...")
    
    for i, (partner_code, store_code) in enumerate(store_list, 1):
        store_key = f"{partner_code}_{store_code}"
        
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(store_list)} stores")
        
        prediction_result = make_prediction_improved(
            input_data=input_data,
            partner_code=partner_code,
            store_code=store_code,
            forecast_days=forecast_days
        )
        
        results[store_key] = prediction_result
        
        if prediction_result['status'] == 'success':
            successful_predictions += 1
    
    # Resumen global
    success_rate = successful_predictions / len(store_list) * 100
    
    print(f"✅ Bulk predictions completed:")
    print(f"  - Successful: {successful_predictions}/{len(store_list)} ({success_rate:.1f}%)")
    
    return {
        'store_predictions': results,
        'summary': {
            'total_stores': len(store_list),
            'successful_predictions': successful_predictions,
            'success_rate_pct': success_rate,
            'forecast_days': forecast_days
        },
        'version': _version,
        'status': 'completed'
    }
