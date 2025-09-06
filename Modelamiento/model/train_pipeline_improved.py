import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

from config.core import config
from pipeline_improved import lifemiles_improved_preprocessing_pipe, lifemiles_improved_forecast_pipe
from processing.data_manager import load_dataset, save_pipeline, save_store_models
from processing.features_improved import get_improved_feature_columns, TargetTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """Calculate SMAPE metric."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator = np.where(denominator == 0, 1e-8, denominator)
    
    smape = 100 * np.mean(np.abs(y_true - y_pred) / denominator)
    return smape


def create_features_for_store(data: pd.DataFrame) -> pd.DataFrame:
    """Create improved features for a specific store's data."""
    
    # Aplicar pipeline de preprocesamiento mejorado
    processed_data = lifemiles_improved_preprocessing_pipe.fit_transform(data)
    
    return processed_data


def train_store_model_improved(store_data: pd.DataFrame, store_code: str, partner_code: str) -> Dict:
    """Train an improved Random Forest model for a specific store."""
    
    try:
        print(f"Training improved model for Store {store_code}, Partner {partner_code}")
        
        # Obtener columnas de características mejoradas
        feature_columns = get_improved_feature_columns()
        
        # Excluir columnas categóricas y no numéricas
        exclude_columns = ['partner_code', 'partner_name', 'store_code', 'store_name', 'date', config.model_config.target]
        numeric_features = [col for col in feature_columns if col not in exclude_columns]
        
        # Filtrar solo las columnas que existen
        available_features = [col for col in numeric_features if col in store_data.columns]
        
        if len(available_features) == 0:
            print(f"No numeric features available for Store {store_code}, Partner {partner_code}")
            return None
        
        print(f"Using {len(available_features)} numeric features for training")
        
        # Preparar X y y
        X = store_data[available_features].copy()
        y = store_data[config.model_config.target].copy()
        
        # Asegurar que todas las columnas son numéricas
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                except:
                    X = X.drop(columns=[col])
                    print(f"Dropped non-numeric column: {col}")
        
        # Remover filas con NaN en las características
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        if len(X) < 50:  # Mínimo de datos requeridos
            print(f"Insufficient data for Store {store_code}, Partner {partner_code}: {len(X)} samples")
            return None
        
        # Split temporal (80/20)
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        # Normalizar características
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Entrenar modelo con configuración mejorada
        model = lifemiles_improved_forecast_pipe
        model.fit(X_train_scaled, y_train)
        
        # Predicciones
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Si se usó transformación de target, aplicar transformación inversa
        if hasattr(config.model_config, 'use_target_transformation') and config.model_config.use_target_transformation:
            target_transformer = TargetTransformer()
            y_train_original = target_transformer.inverse_transform(y_train)
            y_test_original = target_transformer.inverse_transform(y_test)
            y_train_pred_original = target_transformer.inverse_transform(y_train_pred)
            y_test_pred_original = target_transformer.inverse_transform(y_test_pred)
        else:
            y_train_original = y_train
            y_test_original = y_test
            y_train_pred_original = y_train_pred
            y_test_pred_original = y_test_pred
        
        # Métricas de evaluación
        train_mae = mean_absolute_error(y_train_original, y_train_pred_original)
        train_rmse = np.sqrt(mean_squared_error(y_train_original, y_train_pred_original))
        train_r2 = r2_score(y_train_original, y_train_pred_original)
        train_smape = symmetric_mean_absolute_percentage_error(y_train_original, y_train_pred_original)
        
        test_mae = mean_absolute_error(y_test_original, y_test_pred_original)
        test_rmse = np.sqrt(mean_squared_error(y_test_original, y_test_pred_original))
        test_r2 = r2_score(y_test_original, y_test_pred_original)
        test_smape = symmetric_mean_absolute_percentage_error(y_test_original, y_test_pred_original)
        
        # Clasificar calidad del modelo
        if test_r2 > 0.8:
            quality = "EXCELENTE"
        elif test_r2 > 0.6:
            quality = "BUENO"
        elif test_r2 > 0.4:
            quality = "REGULAR"
        else:
            quality = "POBRE"
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(available_features, model.feature_importances_))
        else:
            feature_importance = {}
        
        print(f"Model trained - R²: {test_r2:.3f}, SMAPE: {test_smape:.1f}%, Quality: {quality}")
        
        return {
            'store_code': store_code,
            'partner_code': partner_code,
            'model': model,
            'scaler': scaler,
            'feature_columns': available_features,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_metrics': {
                'mae': train_mae,
                'rmse': train_rmse,
                'r2': train_r2,
                'smape': train_smape
            },
            'test_metrics': {
                'mae': test_mae,
                'rmse': test_rmse,
                'r2': test_r2,
                'smape': test_smape
            },
            'quality': quality,
            'feature_importance': feature_importance
        }
        
    except Exception as e:
        print(f"Error training model for Store {store_code}, Partner {partner_code}: {e}")
        return None


def run_training_improved(max_stores: int = None, min_samples: int = 1, verbose: bool = False) -> Dict:
    """
    Execute the improved training pipeline for LifeMiles forecasting.
    
    Args:
        max_stores (int): Maximum number of stores to train (None = all stores)
        min_samples (int): Minimum samples required per store (default: 1 - trains all stores)
        verbose (bool): Enable verbose output (default: False)
    
    Returns:
        Dict: Training results including metrics and model performance
    """
    
    print("STARTING IMPROVED LIFEMILES TRAINING PIPELINE")
    print("=" * 60)
    
    if max_stores:
        print(f"LIMITED TRAINING MODE: {max_stores} stores")
    else:
        print(f"FULL TRAINING MODE: All stores")
    
    if verbose:
        print(f"Verbose mode: ENABLED")
    
    # Cargar datos
    print("Loading data...")
    data = load_dataset(file_name=config.app_config.train_data_file)
    print(f"Data loaded: {data.shape}")
    
    # Aplicar preprocesamiento global
    print("Applying preprocessing...")
    data_processed = create_features_for_store(data)
    print(f"Features created: {data_processed.shape}")
    
    # Obtener tiendas únicas
    stores = data_processed[['partner_code', 'store_code']].drop_duplicates()
    print(f"Total stores available: {len(stores)}")
    
    # Filtrar tiendas con suficientes datos
    store_counts = data_processed.groupby(['partner_code', 'store_code']).size()
    valid_stores = stores.merge(
        store_counts[store_counts >= min_samples].reset_index(),
        on=['partner_code', 'store_code']
    )
    
    print(f"Stores with sufficient data (>={min_samples}): {len(valid_stores)}")
    
    # Limitar número de tiendas si se especifica
    if max_stores and max_stores < len(valid_stores):
        valid_stores = valid_stores.head(max_stores)
        print(f"Training limited to: {len(valid_stores)} stores")
    
    print("Training models...")
    if verbose:
        store_names = [f"{row['partner_code']}_{row['store_code']}" for _, row in valid_stores.iterrows()]
        print(f"Stores to process: {store_names}")
    print()
    
    def train_single_store(row):
        partner_code = row['partner_code']
        store_code = row['store_code']
        store_data = data_processed[
            (data_processed['partner_code'] == partner_code) & 
            (data_processed['store_code'] == store_code)
        ].copy()
        return train_store_model_improved(store_data, store_code, partner_code)
    
    # Entrenar modelos
    results = []
    for _, row in valid_stores.iterrows():
        result = train_single_store(row)
        if result is not None:
            results.append(result)
    
    # Filtrar resultados exitosos
    successful_results = [r for r in results if r is not None]
    
    print(f"\nTRAINING SUMMARY:")
    print(f"  Total stores attempted: {len(valid_stores)}")
    print(f"  Successful models: {len(successful_results)}")
    print(f"  Success rate: {len(successful_results)/len(valid_stores)*100:.1f}%")
    
    if max_stores:
        print(f"  Limited training mode: {max_stores} stores requested")
    
    if verbose and successful_results:
        print(f"\nSUCCESSFUL STORES:")
        for result in successful_results[:10]:  # Mostrar primeras 10
            store_id = f"{result['partner_code']}_{result['store_code']}"
            r2 = result['test_metrics']['r2']
            smape = result['test_metrics']['smape'] 
            quality = result['quality']
            print(f"    {store_id}: R²={r2:.3f}, SMAPE={smape:.1f}%, Quality={quality}")
        if len(successful_results) > 10:
            print(f"    ... and {len(successful_results)-10} more stores")
    
    if successful_results:
        # Estadísticas de calidad
        r2_scores = [r['test_metrics']['r2'] for r in successful_results]
        smape_scores = [r['test_metrics']['smape'] for r in successful_results]
        
        quality_counts = {}
        for r in successful_results:
            quality = r['quality']
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
        
        print(f"\nMODEL QUALITY DISTRIBUTION:")
        for quality, count in quality_counts.items():
            pct = count / len(successful_results) * 100
            print(f"  {quality}: {count} models ({pct:.1f}%)")
        
        print(f"\nPERFORMANCE METRICS:")
        print(f"  Average R²: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")
        print(f"  Average SMAPE: {np.mean(smape_scores):.1f}% ± {np.std(smape_scores):.1f}%")
        print(f"  Best R²: {np.max(r2_scores):.3f}")
        print(f"  Best SMAPE: {np.min(smape_scores):.1f}%")
        
        # Guardar modelos
        print(f"\nSaving models...")
        
        # Adaptar formato para save_store_models
        models_for_saving = []
        for result in successful_results:
            model_data = {
                'store_code': result['store_code'],
                'partner_code': result['partner_code'],
                'model': result['model'],
                'scaler': result['scaler'],
                'mae': result['test_metrics']['mae'],
                'rmse': result['test_metrics']['rmse'],
                'features': result['feature_columns'],
                'train_samples': result['train_samples'],
                'val_samples': result['test_samples']
            }
            models_for_saving.append(model_data)
        
        save_store_models(models_data=models_for_saving)
        
        # Guardar pipeline de preprocesamiento
        save_pipeline(pipeline_to_persist=lifemiles_improved_preprocessing_pipe)
        
        print(f"IMPROVED TRAINING COMPLETED SUCCESSFULLY!")
        
        # Evaluar si se cumplieron las métricas objetivo
        avg_r2 = np.mean(r2_scores)
        avg_smape = np.mean(smape_scores)
        
        target_r2 = getattr(config.model_config, 'target_metrics', {}).get('min_r2_score', 0.8)
        target_smape = getattr(config.model_config, 'target_metrics', {}).get('max_smape_score', 10.0)
        
        print(f"\nTARGET ACHIEVEMENT:")
        r2_achieved = "ACHIEVED" if avg_r2 >= target_r2 else "NOT ACHIEVED"
        smape_achieved = "ACHIEVED" if avg_smape <= target_smape else "NOT ACHIEVED"
        
        print(f"  R² Target ({target_r2:.1f}): {r2_achieved} Achieved {avg_r2:.3f}")
        print(f"  SMAPE Target ({target_smape:.1f}%): {smape_achieved} Achieved {avg_smape:.1f}%")
        
    return {
        'total_stores': len(valid_stores),
        'successful_models': len(successful_results),
        'results': successful_results,
        'pipeline': lifemiles_improved_preprocessing_pipe
    }


if __name__ == "__main__":
    result = run_training_improved()
    print("\n🎉 IMPROVED TRAINING PIPELINE COMPLETED!")
