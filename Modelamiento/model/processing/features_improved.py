
"""
Módulo de feature engineering mejorado basado en resultados del test con 2 tiendas.
Mejoras implementadas:
- Manejo de outliers con IQR
- Transformación logarítmica de variable objetivo  
- 28 características avanzadas nuevas
- Limpieza robusta de valores infinitos y NaN
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

def remove_outliers_iqr(df: pd.DataFrame, 
                       target_col: str = 'billings', 
                       factor: float = 2.0) -> pd.DataFrame:
    """
    Remover outliers usando método IQR mejorado.

    Args:
        df: DataFrame con los datos
        target_col: Columna objetivo
        factor: Factor para IQR (2.0 más conservador que 1.5)

    Returns:
        DataFrame sin outliers
    """
    Q1 = df[target_col].quantile(0.25)
    Q3 = df[target_col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR

    return df[(df[target_col] >= lower_bound) & (df[target_col] <= upper_bound)]

def preprocess_target_variable(df: pd.DataFrame, 
                             target_col: str = 'billings',
                             method: str = 'log_plus_one') -> pd.DataFrame:
    """
    Preprocessar variable objetivo para manejar distribución sesgada.

    Args:
        df: DataFrame con los datos
        target_col: Columna objetivo
        method: Método de transformación

    Returns:
        DataFrame con variable transformada
    """
    df = df.copy()

    if method == 'log_plus_one':
        df[f'{target_col}_original'] = df[target_col]
        safe_values = np.maximum(df[target_col], 0)
        df[target_col] = np.log1p(safe_values)

    return df

def inverse_transform_target(values: np.ndarray, method: str = 'log_plus_one') -> np.ndarray:
    """
    Aplicar transformación inversa a las predicciones.

    Args:
        values: Valores transformados
        method: Método usado originalmente

    Returns:
        Valores en escala original
    """
    if method == 'log_plus_one':
        return np.expm1(values)
    return values

def create_advanced_features(df: pd.DataFrame, 
                           target_col: str = 'billings',
                           date_col: str = 'date') -> pd.DataFrame:
    """
    Crear características avanzadas mejoradas.

    Args:
        df: DataFrame con datos ordenados por fecha
        target_col: Columna objetivo
        date_col: Columna de fecha

    Returns:
        DataFrame con características avanzadas
    """
    df = df.copy()
    
    # Asegurar que date es datetime y está ordenado
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    # 0. CREAR CARACTERÍSTICAS TEMPORALES BÁSICAS PRIMERO
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['dayofweek'] = df[date_col].dt.dayofweek  # 0=Monday, 6=Sunday
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['day_of_month'] = df[date_col].dt.day

    # 1. CARACTERÍSTICAS TEMPORALES AVANZADAS
    df['quarter'] = df[date_col].dt.quarter
    df['week_of_year'] = df[date_col].dt.isocalendar().week
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_month_start'] = (df['day_of_month'] <= 5).astype(int)
    df['is_month_end'] = (df['day_of_month'] >= 25).astype(int)

    # 2. CARACTERÍSTICAS DE TENDENCIA
    for window in [3, 7, 14, 30]:
        rolling_mean = df[target_col].rolling(window=window).mean()
        df[f'{target_col}_trend_{window}'] = df[target_col] - rolling_mean

        rolling_std = df[target_col].rolling(window=window).std()
        safe_mean = rolling_mean.replace(0, 1e-8)
        df[f'{target_col}_volatility_{window}'] = rolling_std / safe_mean

    # 3. CARACTERÍSTICAS DE MOMENTUM
    for lag in [1, 3, 7]:
        df[f'{target_col}_momentum_{lag}'] = df[target_col] - df[target_col].shift(lag)
        lagged_values = df[target_col].shift(lag).replace(0, 1e-8)
        df[f'{target_col}_momentum_pct_{lag}'] = (df[target_col] - df[target_col].shift(lag)) / lagged_values * 100

    # 4. CARACTERÍSTICAS DE RANK/PERCENTIL
    for window in [7, 30]:
        df[f'{target_col}_rank_{window}'] = df[target_col].rolling(window=window).rank(pct=True)

    # 5. CARACTERÍSTICAS DE AUTOCORRELACIÓN
    for lag in [7, 14, 30]:
        correlation = df[target_col].rolling(window=30).corr(df[target_col].shift(lag))
        df[f'{target_col}_autocorr_{lag}'] = correlation

    # 6. CARACTERÍSTICAS DE OUTLIERS
    for window in [7, 30]:
        rolling_mean = df[target_col].rolling(window=window).mean()
        rolling_std = df[target_col].rolling(window=window).std()
        safe_std = rolling_std.replace(0, 1e-8)
        df[f'{target_col}_zscore_{window}'] = (df[target_col] - rolling_mean) / safe_std
        df[f'{target_col}_is_outlier_{window}'] = (np.abs(df[f'{target_col}_zscore_{window}']) > 2).astype(int)

    return df

def clean_features(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """
    Limpiar características para remover infinitos y NaN.

    Args:
        df: DataFrame con las características
        feature_columns: Lista de columnas de características

    Returns:
        DataFrame limpio
    """
    df_clean = df.copy()

    # Reemplazar infinitos con NaN
    df_clean[feature_columns] = df_clean[feature_columns].replace([np.inf, -np.inf], np.nan)

    # Rellenar NaN con la mediana de cada columna
    for col in feature_columns:
        if df_clean[col].isna().any():
            median_val = df_clean[col].median()
            if pd.isna(median_val):
                median_val = 0
            df_clean[col] = df_clean[col].fillna(median_val)

    # Verificación final
    df_clean[feature_columns] = df_clean[feature_columns].replace([np.inf, -np.inf], 0)

    return df_clean

def get_all_feature_columns() -> List[str]:
    """
    Obtener lista completa de características para el modelo mejorado.

    Returns:
        Lista de nombres de características
    """
    # Características básicas
    basic_features = ['year', 'month', 'day_of_week', 'day_of_month']

    # Características lag
    lag_features = [f'billings_lag_{lag}' for lag in [1, 2, 3, 7, 14, 30]]

    # Características rolling
    rolling_features = []
    for window in [7, 30]:
        rolling_features.extend([
            f'billings_rolling_{window}_mean',
            f'billings_rolling_{window}_std'
        ])

    # Características avanzadas
    advanced_features = []

    # Temporales avanzadas
    advanced_features.extend([
        'quarter', 'week_of_year', 'is_weekend', 'is_month_start', 'is_month_end'
    ])

    # Tendencia y volatilidad
    for window in [3, 7, 14, 30]:
        advanced_features.extend([
            f'billings_trend_{window}',
            f'billings_volatility_{window}'
        ])

    # Momentum
    for lag in [1, 3, 7]:
        advanced_features.extend([
            f'billings_momentum_{lag}',
            f'billings_momentum_pct_{lag}'
        ])

    # Rank
    for window in [7, 30]:
        advanced_features.append(f'billings_rank_{window}')

    # Autocorrelación
    for lag in [7, 14, 30]:
        advanced_features.append(f'billings_autocorr_{lag}')

    # Outliers
    for window in [7, 30]:
        advanced_features.extend([
            f'billings_zscore_{window}',
            f'billings_is_outlier_{window}'
        ])

    return basic_features + lag_features + rolling_features + advanced_features


def create_all_features_improved(df: pd.DataFrame, 
                               target_col: str = 'billings',
                               date_col: str = 'date') -> pd.DataFrame:
    """
    Crear todas las características mejoradas en un solo paso.
    
    Args:
        df: DataFrame con los datos
        target_col: Nombre de la columna objetivo
        date_col: Nombre de la columna de fecha
    
    Returns:
        DataFrame con todas las características creadas
    """
    # Preprocesar variable objetivo
    df_processed = preprocess_target_variable(df, target_col, method='log_plus_one')
    
    # Crear características avanzadas
    df_features = create_advanced_features(df_processed, target_col, date_col)
    
    return df_features


def get_improved_feature_columns() -> List[str]:
    """
    Obtener la lista de todas las columnas de características mejoradas.
    
    Returns:
        Lista de nombres de columnas de características (incluye categóricas y numéricas)
    """
    # Características básicas (incluye categóricas)
    basic_features = ['partner_code', 'partner_name', 'store_code', 'store_name', 'date', 'billings', 'miles']
    
    # Características temporales básicas
    temporal_features = ['year', 'month', 'day', 'dayofweek', 'quarter', 'week_of_year', 'is_weekend', 'is_month_start', 'is_month_end']
    
    # Lag features
    lag_features = []
    for lag in [1, 2, 3, 7, 14, 30]:
        lag_features.append(f'billings_lag_{lag}')
    
    # Rolling features  
    rolling_features = []
    for window in [3, 7, 14, 30]:
        rolling_features.extend([
            f'billings_rolling_{window}_mean',
            f'billings_rolling_{window}_std', 
            f'billings_rolling_{window}_min',
            f'billings_rolling_{window}_max'
        ])
    
    # Advanced features
    advanced_features = []
    
    # Trend
    for window in [3, 7, 14, 30]:
        advanced_features.append(f'billings_trend_{window}')
    
    # Volatility  
    for window in [7, 30]:
        advanced_features.append(f'billings_volatility_{window}')
    
    # Momentum
    for lag in [1, 3, 7]:
        advanced_features.extend([
            f'billings_momentum_{lag}',
            f'billings_momentum_pct_{lag}'
        ])
    
    # Rank
    for window in [7, 30]:
        advanced_features.append(f'billings_rank_{window}')
    
    # Autocorrelación
    for lag in [7, 14, 30]:
        advanced_features.append(f'billings_autocorr_{lag}')
    
    # Outliers
    for window in [7, 30]:
        advanced_features.extend([
            f'billings_zscore_{window}',
            f'billings_is_outlier_{window}'
        ])
    
    return basic_features + temporal_features + lag_features + rolling_features + advanced_features


def create_features_for_store(data: pd.DataFrame) -> pd.DataFrame:
    """
    Crear características para todas las tiendas en el dataset.
    
    Args:
        data: DataFrame con datos de todas las tiendas
    
    Returns:
        DataFrame con características creadas para todas las tiendas
    """
    results = []
    
    # Procesar por tienda
    for (partner_code, store_code), store_data in data.groupby(['partner_code', 'store_code']):
        try:
            # Crear características para esta tienda
            store_features = create_all_features_improved(store_data.copy())
            
            # Mantener identificadores de tienda
            store_features['partner_code'] = partner_code
            store_features['store_code'] = store_code
            
            results.append(store_features)
            
        except Exception as e:
            print(f"Warning: Error processing store {partner_code}_{store_code}: {e}")
            continue
    
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame()


# =====================================================
# CLASES PARA PIPELINE
# =====================================================

from sklearn.base import BaseEstimator, TransformerMixin

class ImprovedOutlierRemover(BaseEstimator, TransformerMixin):
    """
    Transformer para remover outliers usando método IQR mejorado.
    """
    
    def __init__(self, target_col: str = 'billings', factor: float = 2.0):
        self.target_col = target_col
        self.factor = factor
        self.lower_bound_ = None
        self.upper_bound_ = None
    
    def fit(self, X, y=None):
        """Calcular los bounds para outliers."""
        if self.target_col in X.columns:
            Q1 = X[self.target_col].quantile(0.25)
            Q3 = X[self.target_col].quantile(0.75)
            IQR = Q3 - Q1
            
            self.lower_bound_ = Q1 - self.factor * IQR
            self.upper_bound_ = Q3 + self.factor * IQR
        
        return self
    
    def transform(self, X):
        """Remover outliers del DataFrame."""
        if self.target_col in X.columns and self.lower_bound_ is not None:
            mask = (X[self.target_col] >= self.lower_bound_) & (X[self.target_col] <= self.upper_bound_)
            return X[mask].copy()
        return X.copy()


class TargetTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer para preprocessar la variable objetivo.
    """
    
    def __init__(self, target_col: str = 'billings', method: str = 'log_plus_one'):
        self.target_col = target_col
        self.method = method
        self.min_value_ = None
    
    def fit(self, X, y=None):
        """Fit del transformer."""
        if self.target_col in X.columns:
            self.min_value_ = X[self.target_col].min()
        return self
    
    def transform(self, X):
        """Transformar la variable objetivo."""
        X_transformed = X.copy()
        
        if self.target_col in X_transformed.columns:
            if self.method == 'log_plus_one':
                # Asegurar valores positivos
                X_transformed[self.target_col] = np.maximum(X_transformed[self.target_col], 0.1)
                X_transformed[f'{self.target_col}_log'] = np.log1p(X_transformed[self.target_col])
            elif self.method == 'sqrt':
                X_transformed[f'{self.target_col}_sqrt'] = np.sqrt(np.maximum(X_transformed[self.target_col], 0))
        
        return X_transformed
    
    def inverse_transform(self, X):
        """Transformación inversa."""
        X_inverse = X.copy()
        
        if self.method == 'log_plus_one':
            if f'{self.target_col}_log' in X_inverse.columns:
                X_inverse[self.target_col] = np.expm1(X_inverse[f'{self.target_col}_log'])
        elif self.method == 'sqrt':
            if f'{self.target_col}_sqrt' in X_inverse.columns:
                X_inverse[self.target_col] = X_inverse[f'{self.target_col}_sqrt'] ** 2
        
        return X_inverse


class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformer para crear características avanzadas mejoradas.
    """
    
    def __init__(self, target_col: str = 'billings', date_col: str = 'date'):
        self.target_col = target_col
        self.date_col = date_col
        self.feature_names_ = None
    
    def fit(self, X, y=None):
        """Fit del transformer."""
        # Las características se calculan en transform
        return self
    
    def transform(self, X):
        """Crear todas las características avanzadas."""
        X_transformed = X.copy()
        
        # Asegurar que date es datetime
        if self.date_col in X_transformed.columns:
            X_transformed[self.date_col] = pd.to_datetime(X_transformed[self.date_col])
            X_transformed = X_transformed.sort_values(self.date_col)
        
        # Crear todas las características
        X_transformed = create_all_features_improved(X_transformed, self.target_col, self.date_col)
        
        # Limpiar valores infinitos y NaN
        X_transformed = X_transformed.replace([np.inf, -np.inf], np.nan)
        X_transformed = X_transformed.bfill().ffill().fillna(0)
        
        # Guardar nombres de características
        self.feature_names_ = X_transformed.columns.tolist()
        
        return X_transformed
    
    def get_feature_names_out(self, input_features=None):
        """Obtener nombres de características de salida."""
        if self.feature_names_ is not None:
            return self.feature_names_
        return input_features
