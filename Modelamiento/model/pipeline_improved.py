"""
Pipeline mejorado para el modelo LifeMiles basado en resultados del test exitoso.
Incorpora todas las mejoras: outlier removal, transformación de target, 
características avanzadas y modelo optimizado.
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from processing.features_improved import (
    ImprovedOutlierRemover,
    TargetTransformer, 
    AdvancedFeatureEngineer,
    create_all_features_improved,
    remove_outliers_iqr,
    preprocess_target_variable,
    create_advanced_features
)
from config.core import config


# Pipeline de preprocesamiento mejorado
lifemiles_improved_preprocessing_pipe = Pipeline([
    # 1. Remover outliers usando IQR mejorado
    ('outlier_remover', ImprovedOutlierRemover(
        target_col=config.model_config.target,
        factor=getattr(config.model_config, 'outlier_factor', 2.0)
    )),
    
    # 2. Transformar variable objetivo 
    ('target_transformer', TargetTransformer(
        target_col=config.model_config.target,
        method='log_plus_one'
    )),
    
    # 3. Crear todas las características avanzadas
    ('advanced_features', AdvancedFeatureEngineer(
        target_col=config.model_config.target,
        date_col='date'
    ))
])


# Pipeline de modelo mejorado 
lifemiles_improved_forecast_pipe = RandomForestRegressor(
    n_estimators=getattr(config.model_config, 'n_estimators', 200),
    max_depth=getattr(config.model_config, 'max_depth', 15),
    min_samples_split=getattr(config.model_config, 'min_samples_split', 5),
    min_samples_leaf=getattr(config.model_config, 'min_samples_leaf', 2),
    max_features=getattr(config.model_config, 'max_features', 'sqrt'),
    random_state=getattr(config.model_config, 'random_state', 42),
    n_jobs=-1
)
