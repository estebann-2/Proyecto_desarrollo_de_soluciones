# MLflow Support for LifeMiles Forecasting

Este documento describe cómo se implementó el soporte de MLflow en el proyecto de forecasting de LifeMiles y cómo utilizarlo.

## ¿Qué es MLflow y por qué lo usamos?

MLflow es una plataforma open-source para el ciclo de vida de machine learning que nos permite:

- **Tracking**: Registrar experimentos, parámetros, métricas y artefactos
- **Reproducibilidad**: Mantener un historial completo de entrenamientos
- **Comparación**: Comparar diferentes versiones de modelos fácilmente
- **Deployment**: Facilitar el deployment de modelos a producción

## Implementación del Soporte MLflow

### 1. Ubicación de la Implementación

El soporte de MLflow se implementó en los siguientes archivos:

```
Modelamiento/
├── model/
│   ├── config/
│   │   └── mlflow_config.py          # Configuración de MLflow
│   └── train_pipeline_improved.py    # Logging de entrenamiento
├── mlflow_explorer.py                # Herramienta de exploración
└── mlruns/                          # Directorio de experimentos (se crea automáticamente)
```

### 2. Configuración Principal

**Archivo: `model/config/mlflow_config.py`**

Este archivo contiene:
- Configuración del tracking URI
- Creación automática de experimentos
- Funciones de inicialización
- Configuración por defecto

### 3. Logging en el Entrenamiento

**Archivo: `model/train_pipeline_improved.py`**

Se agregó logging para:

#### Experimento Principal:
- Parámetros del pipeline (max_stores, min_samples, etc.)
- Información del dataset
- Métricas agregadas (R² promedio, SMAPE promedio)
- Distribución de calidad de modelos
- Tasa de éxito del entrenamiento

#### Modelos Individuales (nested runs):
- Parámetros específicos del modelo (store_code, partner_code, n_features)
- Métricas de entrenamiento y validación
- Calidad del modelo
- Artefactos (modelo, scaler, feature importance)
- Tags para organización

## Métricas Registradas

### Experimento Principal
- `total_stores_attempted`: Número total de tiendas procesadas
- `successful_models`: Número de modelos entrenados exitosamente
- `success_rate`: Porcentaje de éxito del entrenamiento
- `avg_r2`: R² promedio de todos los modelos
- `avg_smape`: SMAPE promedio de todos los modelos
- `std_r2`: Desviación estándar del R²
- `std_smape`: Desviación estándar del SMAPE
- `best_r2`: Mejor R² obtenido
- `best_smape`: Mejor SMAPE obtenido
- `quality_*_count`: Conteo de modelos por calidad
- `quality_*_pct`: Porcentaje de modelos por calidad

### Modelos Individuales
- `train_mae`, `train_rmse`, `train_r2`, `train_smape`: Métricas de entrenamiento
- `test_mae`, `test_rmse`, `test_r2`, `test_smape`: Métricas de validación
- `n_features`: Número de características utilizadas
- `train_samples`: Número de muestras de entrenamiento
- `test_samples`: Número de muestras de validación

## Cómo Usar MLflow

### 1. Entrenar Modelos con MLflow

```bash
# Entrenamiento normal - se registra automáticamente en MLflow
python run_improved_training.py

# Entrenamiento limitado - también se registra
python run_improved_training.py --n 5
```

### 2. Explorar Experimentos

```bash
# Listar todos los experimentos
python mlflow_explorer.py

# Ver runs de un experimento específico
python mlflow_explorer.py --experiment-id 1

# Ver los mejores modelos
python mlflow_explorer.py --best-models

# Iniciar la interfaz web de MLflow
python mlflow_explorer.py --serve
```

### 3. Interfaz Web de MLflow

Para acceder a la interfaz web completa:

```bash
# Opción 1: Usar el explorer
python mlflow_explorer.py --serve

# Opción 2: Comando directo
cd Modelamiento/
mlflow ui --backend-store-uri file://./mlruns
```

Luego abrir http://localhost:5000 en el navegador.

## Visualizaciones Disponibles

En la interfaz de MLflow puedes:

1. **Comparar Experimentos**: Ver todos los entrenamientos realizados
2. **Métricas por Tiempo**: Evolución de métricas a lo largo del tiempo
3. **Gráficos de Scatter**: Comparar métricas entre diferentes runs
4. **Artefactos**: Descargar modelos, scalers y análisis de feature importance
5. **Filtros y Búsquedas**: Buscar modelos específicos por calidad, métricas, etc.

## Beneficios de la Implementación

### Para el Desarrollo
- **Trazabilidad**: Historial completo de todos los entrenamientos
- **Debugging**: Identificar qué configuraciones funcionan mejor
- **Iteración**: Comparar mejoras entre versiones

### Para la Evaluación 
- **Soporte de MLFlow**: Implementado completamente
- **Tracking de Experimentos**: Todos los entrenamientos se registran
- **Métricas Apropiadas**: R², SMAPE, MAE, RMSE registradas
- **Reproducibilidad**: Parámetros y configuraciones guardadas

### Para Producción
- **Registro de Modelos**: Fácil identificación del mejor modelo
- **Versionado**: Control de versiones automático
- **Deployment**: Integración sencilla con sistemas de deployment

## Estructura de Experimentos

```
LifeMiles_Forecasting_Training/
├── Run 1: Training_Pipeline_20240906_143022
│   ├── Métricas del pipeline completo
│   ├── Nested Run 1.1: Store ADICO_10922920
│   ├── Nested Run 1.2: Store ADICO_11426079
│   └── ...
├── Run 2: Training_Pipeline_20240906_150815
│   └── ...
```

## Configuración Avanzada

### Cambiar Ubicación de MLruns

Editar `model/config/mlflow_config.py`:

```python
def setup_mlflow_tracking():
    # Cambiar esta línea para usar otra ubicación
    mlruns_dir = "/ruta/personalizada/mlruns"
    mlflow.set_tracking_uri(f"file://{mlruns_dir}")
```

### Agregar Nuevas Métricas

En `train_pipeline_improved.py`, agregar en la sección de MLflow logging:

```python
# Agregar nueva métrica
mlflow.log_metric("nueva_metrica", valor)
```

## Consideraciones Importantes

1. **Espacio en Disco**: Los experimentos pueden ocupar espacio considerable
2. **Performance**: El logging agrega overhead mínimo al entrenamiento
3. **Backup**: Considerar respaldar el directorio `mlruns/` regularmente
4. **Limpieza**: Eliminar experimentos antiguos según sea necesario

## Troubleshooting

### Error: "No module named 'mlflow'"
```bash
pip install mlflow>=2.8.0
```

### MLflow UI no inicia
Verificar que estés en el directorio correcto y que `mlruns/` exista:
```bash
cd Modelamiento/
ls -la mlruns/
```

### No aparecen experimentos
Verificar que se ejecutó al menos un entrenamiento después de implementar MLflow.

## Métricas de Éxito del Soporte MLflow

El soporte de MLflow se considera exitoso cuando:

- Todos los entrenamientos se registran automáticamente
- Las métricas se pueden comparar entre runs
- Los artefactos (modelos) se guardan correctamente
- La interfaz web es accesible y funcional
- Se puede identificar fácilmente el mejor modelo

---

**Implementado por**: Sistema de Entrenamiento LifeMiles  
**Fecha**: Septiembre 2024  
**Versión MLflow**: 2.8.0+
