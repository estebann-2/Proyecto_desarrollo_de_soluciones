# Documentación del Módulo de Modelamiento - LifeMiles

## Descripción General

El módulo de **Modelamiento** es el núcleo del sistema de predicción de millas LifeMiles. Implementa un pipeline completo de machine learning para el forecasting de millas por tienda, utilizando Random Forest con características avanzadas de ingeniería de features y integración con MLflow para tracking de experimentos.

## Arquitectura del Sistema

```
Modelamiento/
├── Core del Modelo
│   ├── model/                          # Módulo principal del modelo
│   │   ├── __init__.py
│   │   ├── VERSION                     # Control de versiones
│   │   ├── config.yml                  # Configuración base
│   │   ├── pipeline_improved.py        # Pipeline de preprocessing
│   │   ├── predict_improved.py         # Módulo de predicción
│   │   └── train_pipeline_improved.py  # Pipeline de entrenamiento
│   │
│   ├── Configuraciones
│   │   ├── config/
│   │   │   ├── __init__.py
│   │   │   ├── core.py                 # Configuración central
│   │   │   ├── improved_config.yml     # Configuración del modelo mejorado
│   │   │   └── mlflow_config.py        # Configuración de MLflow
│   │
│   ├── Datos
│   │   ├── datasets/
│   │   │   ├── __init__.py
│   │   │   └── data_lifemiles.csv     # Dataset principal
│   │
│   ├── Procesamiento
│   │   ├── processing/
│   │   │   ├── __init__.py
│   │   │   ├── data_manager.py         # Gestión de datos
│   │   │   ├── features_improved.py    # Ingeniería de features
│   │   │   └── validation.py           # Validación de datos
│   │
│   └── Modelos Entrenados
│       └── trained/
│           ├── __init__.py
│           ├── modelo-lifemiles-forecast-improved-output_preprocessing_0.0.1.pkl
│           └── store_models/           # Modelos por tienda
│
├── Scripts de Ejecución
│   ├── run_improved_training.py        # Entrenamiento principal
│   ├── run_improved_predictions.py     # Predicciones mejoradas
│   ├── generate_predictions.py         # Generación de predicciones futuras
│   ├── show_predictions_table.py       # Visualización de predicciones
│   └── test_predictions.py             # Testing de predicciones
│
├── MLflow y Tracking
│   ├── mlflow_explorer.py              # Explorador de experimentos
│   ├── MLflow_README.md                # Documentación de MLflow
│   └── mlruns/                         # Experimentos y artefactos
│
├── Configuración del Proyecto
│   ├── requirements/
│   │   ├── requirements.txt            # Dependencias principales
│   │   ├── test_requirements.txt       # Dependencias de testing
│   │   └── typing_requirements.txt     # Dependencias de typing
│   ├── pyproject.toml                  # Configuración de proyecto
│   ├── setup.py                        # Setup del paquete
│   └── MANIFEST.in                     # Archivos incluidos en distribución
│
├── Testing
│   └── tests/
│       └── test_prediction.py          # Tests unitarios
│
├── Outputs
│   └── predicciones_futuras.csv        # Archivo de predicciones generadas
│
└── Infraestructura
    └── llavep2.pem                     # Clave SSH para VM
```

## Componentes Principales

### 1. Core del Modelo (`model/`)

#### **Pipeline de Preprocessing (`pipeline_improved.py`)**
- Pipeline de scikit-learn para preprocessing automático
- Manejo de valores faltantes
- Codificación de variables categóricas
- Normalización y escalado de features

#### **Entrenamiento (`train_pipeline_improved.py`)**
- Pipeline completo de entrenamiento
- Entrenamiento por tienda individual
- Validación cruzada con Time Series Split
- Métricas: R², SMAPE, MAE, RMSE
- Integración con MLflow para tracking
- Gestión automática de modelos por tienda

#### **Predicción (`predict_improved.py`)**
- Carga automática de modelos entrenados
- Predicciones individuales por tienda
- Manejo de errores y fallbacks
- Validación de datos de entrada

### 2. Ingeniería de Features (`processing/features_improved.py`)

El sistema implementa **42 características avanzadas** organizadas en categorías:

#### **Features Temporales Básicas (8)**
- Año, mes, día, día de la semana
- Trimestre, número de semana
- Indicadores de fin de semana
- Días desde el inicio del dataset

#### **Features Temporales Avanzadas (6)**
- Trigonométricas (sin/cos) para mes y día de la semana
- Indicadores de días especiales (inicio/fin de mes/año)
- Patrones cíclicos estacionales

#### **Lag Features (6)**
- Valores históricos (lag 1, 7, 14, 30 días)
- Promedios móviles de diferentes ventanas
- Tendencias a corto y mediano plazo

#### **Features de Volatilidad y Tendencia (7)**
- Desviación estándar móvil
- Coeficiente de variación
- Tendencias lineales por ventanas
- Momentum y aceleración

#### **Features de Ranking y Percentiles (8)**
- Rankings por partner y globales
- Percentiles de millas por período
- Posiciones relativas en distribuciones

#### **Features de Autocorrelación (7)**
- Autocorrelaciones con diferentes lags
- Detección de patrones estacionales
- Medidas de persistencia temporal

### 3. MLflow Integration

#### **Tracking de Experimentos**
- **Experimento Principal**: `LifeMiles_Forecasting_Training`
- **Métricas Registradas**: R², SMAPE, MAE, RMSE
- **Parámetros**: Configuración del modelo, preprocessing
- **Artefactos**: Modelos entrenados, gráficos de performance

#### **Logging Automático**
- Cada entrenamiento crea un run en MLflow
- Métricas agregadas del pipeline completo
- Distribución de calidad de modelos
- Comparación de objetivos vs resultados

### 4. Scripts de Ejecución

#### **Entrenamiento (`run_improved_training.py`)**
```bash
# Entrenamiento completo
python run_improved_training.py

# Entrenamiento limitado (testing)
python run_improved_training.py --n 5

# Con verbosidad
python run_improved_training.py --verbose
```

**Funcionalidades:**
- Entrenamiento masivo o limitado por número de tiendas
- Validación automática de datos
- Reporte detallado de performance
- Guardado automático de modelos

#### **Predicciones (`run_improved_predictions.py`)**
```bash
# Predicciones para datos existentes
python run_improved_predictions.py

# Para tienda específica
python run_improved_predictions.py --store COBU3_11475795
```

#### **Predicciones Futuras (`generate_predictions.py`)**
```bash
# Próximos 7 días (default)
python generate_predictions.py

# Próximos 30 días
python generate_predictions.py --days 30

# Desde fecha específica
python generate_predictions.py --start-date 2024-01-01
```

### 5. Configuración del Modelo

#### **Algoritmo Principal: Random Forest Mejorado**
```yaml
model_config:
  algorithm_name: 'random_forest_improved'
  n_estimators: 200        # Incrementado para mejor performance
  max_depth: 15            # Balanceado para evitar overfitting
  min_samples_split: 5     # Control de complejidad
  min_samples_leaf: 2      # Regularización
  max_features: 'sqrt'     # Reducción de varianza
  random_state: 42
  n_jobs: -1              # Paralelización completa
```

#### **Preprocessing Avanzado**
```yaml
preprocessing:
  remove_outliers: true
  outlier_method: 'iqr'
  iqr_factor: 2.0
  transform_target: true
  target_transformation: 'log_plus_one'
```

## Métricas y Performance

### **Objetivos del Modelo**
- **R² Target**: ≥ 0.8 (Excelente predictibilidad)
- **SMAPE Target**: ≤ 10% (Error promedio bajo)

### **Clasificación de Calidad**
- **EXCELENTE**: R² ≥ 0.8 y SMAPE ≤ 10%
- **BUENO**: R² ≥ 0.6 y SMAPE ≤ 20%
- **REGULAR**: R² ≥ 0.4 y SMAPE ≤ 30%
- **POBRE**: Por debajo de los umbrales anteriores

### **Métricas Reportadas**
- **R² (Coeficiente de Determinación)**: Varianza explicada
- **SMAPE (Symmetric Mean Absolute Percentage Error)**: Error porcentual
- **MAE (Mean Absolute Error)**: Error absoluto promedio
- **RMSE (Root Mean Square Error)**: Error cuadrático medio

## Flujo de Trabajo

### **1. Entrenamiento**
```
Datos → Features Engineering → Preprocessing → Training → Validation → Model Storage → MLflow Logging
```

### **2. Predicción**
```
Nuevos Datos → Features Engineering → Preprocessing → Modelo Cargado → Predicción → Validación → Output
```

### **3. Predicciones Futuras**
```
Fecha Objetivo → Generación de Features → Modelos por Tienda → Agregación → CSV Output
```

## Dependencias Principales

### **Core ML Stack**
- **scikit-learn**: Algoritmos de ML y pipeline
- **pandas**: Manipulación de datos
- **numpy**: Computación numérica

### **Tracking y Experimentación**
- **mlflow**: Tracking de experimentos y modelos
- **matplotlib/seaborn**: Visualizaciones
- **plotly**: Gráficos interactivos

### **Utilities**
- **joblib**: Serialización de modelos
- **pydantic**: Validación de configuraciones
- **strictyaml**: Configuraciones YAML seguras

## Testing y Validación

### **Tests Automatizados**
- Tests unitarios para predicciones
- Validación de pipeline completo
- Tests de integración con MLflow

### **Validación de Modelos**
- Time Series Cross Validation
- Validación por tienda individual
- Análisis de residuos y distribuciones

## Archivos de Salida

### **Modelos Entrenados**
- `modelo-lifemiles-forecast-improved-*.pkl`: Pipeline de preprocessing
- `store_models/model_*.pkl`: Modelos individuales por tienda

### **Predicciones**
- `predicciones_futuras.csv`: Predicciones futuras generadas
- Archivos temporales de validación en testing

### **MLflow Artifacts**
- Gráficos de performance por modelo
- Métricas de validación
- Configuraciones de entrenamiento

## Uso en Producción

### **Entrenamiento Programado**
1. Ejecutar `run_improved_training.py` periódicamente
2. Monitorear performance en MLflow UI
3. Validar métricas contra objetivos
4. Actualizar modelos en producción

### **Predicciones en Tiempo Real**
1. Usar `run_improved_predictions.py` para datos existentes
2. Usar `generate_predictions.py` para forecasting
3. Integrar con sistemas downstream via CSV

### **Monitoreo**
1. MLflow UI para tracking de experimentos
2. Métricas de performance agregadas
3. Alertas basadas en degradación de R² o aumento de SMAPE

## Configuración del Entorno

### **Instalación**
```bash
pip install -r requirements/requirements.txt
```

### **MLflow UI**
```bash
python mlflow_explorer.py --serve
# Acceso en: http://localhost:5000
```

### **Variables de Entorno**
- MLflow tracking automático en `./mlruns/`
- Configuraciones en archivos YAML
- Logs automáticos en consola y MLflow

---

**Fecha de documentación**: Septiembre 2025  
**Versión del modelo**: 0.0.1  
**Autor**: Sistema LifeMiles ML Pipeline
