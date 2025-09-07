# Proyecto Desarrollo de Soluciones

Proyecto de ciencia de datos enfocado en el desarrollo de soluciones analíticas utilizando técnicas de machine learning y análisis exploratorio de datos.

## 📋 Descripción

Este proyecto implementa un pipeline de ciencia de datos completo, desde el análisis exploratorio hasta el desarrollo de modelos predictivos, utilizando herramientas modernas para el versionado de datos y reproducibilidad de experimentos.

## Breve explicación del código predictivo

Carga de modelos entrenados por tienda.
El script busca archivos en model/trained/store_models/ con el patrón model_<partner>_<store>.pkl y los carga con joblib. Cada archivo representa un modelo ya entrenado para una tienda particular (identificada por <partner_code>_<store_code>).

Definición del horizonte de predicción.
A partir de los argumentos de línea de comandos, se define la fecha de inicio (por defecto, mañana) y el número de días a predecir (por defecto, 7). También se puede limitar a una tienda específica con --store PARTNER_STORE.

Plantilla de datos futuros.
Usando los datos históricos (cargados desde la ruta configurada en config.app_config.train_data_file), se genera una plantilla para las fechas futuras de cada tienda.

Se toma una ventana reciente (p. ej., últimos 60 días) como referencia.

Se extraen estadísticas por día de la semana para el target (típicamente billings), y se generan valores simulados (media por día ± ruido controlado) para alimentar el pipeline de features.

Se actualizan las variables temporales: año, mes, día, trimestre, semana del año, día de la semana, banderas de inicio/fin de mes, etc.

Opcionalmente se simulan otras columnas (p. ej., promocion_activa, precio_promedio, etc.) con pequeñas variaciones.

Preprocesamiento + predicción.
La plantilla se transforma con lifemiles_improved_preprocessing_pipe y, sobre las columnas disponibles definidas en get_improved_feature_columns(), se ejecuta el método .predict() del modelo correspondiente a la tienda. El resultado es un DataFrame con: fecha, partner, tienda y predicted_monto.

Salida y resumen.

Se imprime un resumen en consola por tienda (rango de fechas y totales).

Se consolidan todas las predicciones y se guardan en un CSV (por defecto, predicciones_futuras.csv, modificable con --output).

Se reporta el total de predicciones y tiendas procesadas.

## 🏗️ Estructura del Proyecto

```
Proyecto_desarrollo_de_soluciones/
├── .dvc/                    # Configuración de Data Version Control
├── EDA/                     # Análisis Exploratorio de Datos
│   └── *.ipynb             # Notebooks de análisis
├── .dvcignore              # Archivos ignorados por DVC
├── .gitignore              # Archivos ignorados por Git
├── data.dvc                # Referencias de datos versionados
└── README.md               # Este archivo
```

## 🛠️ Tecnologías Utilizadas

- **Python** - Lenguaje principal de desarrollo
- **Jupyter Notebook** - Desarrollo interactivo y análisis
- **DVC (Data Version Control)** - Versionado de datos y experimentos
- **Git** - Control de versiones del código

## 🚀 Instalación y Configuración

### Prerrequisitos

- Python 3.8+
- Git
- DVC

### Configuración del Entorno

```bash
# Clonar el repositorio
git clone https://github.com/estebann-2/Proyecto_desarrollo_de_soluciones.git
cd Proyecto_desarrollo_de_soluciones

# Instalar DVC
pip install dvc

# Configurar DVC y descargar datos
dvc pull

# Instalar dependencias (si existe requirements.txt)
pip install -r requirements.txt
```

## 📊 Uso

1. **Análisis Exploratorio**: Navegar a la carpeta `EDA/` y ejecutar los notebooks
2. **Datos**: Los datos están versionados con DVC - usar `dvc pull` para obtener la última versión
3. **Experimentos**: Seguir la estructura de notebooks para reproducir análisis

## 📁 Componentes Principales

### EDA (Exploratory Data Analysis)

Contiene notebooks de Jupyter con análisis exploratorio detallado de los datos, incluyendo:

- Limpieza y preprocesamiento de datos
- Visualizaciones y estadísticas descriptivas
- Identificación de patrones y insights

### Control de Versiones de Datos

- **DVC** gestiona las versiones de los datasets
- **Git** controla el código y metadatos
- Reproducibilidad garantizada de experimentos

## 👥 Contribuidores

- **[estebann-2](https://github.com/estebann-2)** - Esteban Caicedo
- **[crdadiaz855-web](https://github.com/crdadiaz855-web)**
- **[jucagu](https://github.com/Jucagu)** - Juan Camilo Gutierrez Diaz

## 🔄 Workflow de Desarrollo

1. Análisis exploratorio en notebooks
2. Versionado de datos con DVC
3. Control de versiones de código con Git
4. Documentación de resultados y metodología

## 📈 Próximos Pasos

- [x] Agregar documentación detallada de metodología
- [x] Implementar pipeline de ML automatizado
- [x] Agregar tests unitarios para funciones de procesamiento
- [x] Crear visualizaciones interactivas

## 📄 Licencia

Este proyecto está bajo la licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 📞 Contacto

Para preguntas o colaboraciones, contactar a los contribuidores del proyecto a través de GitHub.

---

**Desarrollado con 💻 y ☕ por el equipo de Desarrollo de Soluciones**
