# Proyecto Desarrollo de Soluciones

Proyecto de ciencia de datos enfocado en el desarrollo de soluciones analíticas utilizando técnicas de machine learning y análisis exploratorio de datos.

## 📋 Descripción

Este proyecto implementa un pipeline de ciencia de datos completo, desde el análisis exploratorio hasta el desarrollo de modelos predictivos, utilizando herramientas modernas para el versionado de datos y reproducibilidad de experimentos.

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

## 🔄 Workflow de Desarrollo

1. Análisis exploratorio en notebooks
2. Versionado de datos con DVC
3. Control de versiones de código con Git
4. Documentación de resultados y metodología

## 📈 Próximos Pasos

- [ ] Agregar documentación detallada de metodología
- [ ] Implementar pipeline de ML automatizado
- [ ] Agregar tests unitarios para funciones de procesamiento
- [ ] Crear visualizaciones interactivas

## 📄 Licencia

Este proyecto está bajo la licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 📞 Contacto

Para preguntas o colaboraciones, contactar a los contribuidores del proyecto a través de GitHub.

---

**Desarrollado con 💻 y ☕ por el equipo de Desarrollo de Soluciones**
