# AdM-Grupo-8-UCDP-GED

Este repositorio reúne el análisis del dataset "Eventos de violencia organizada (UCDP GED)" y entrenamiento de modelos para el trabajo del grupo 8 de la asignatura de Aprendizaje de Máquina.

## Descripción

El dataset UCDP GED documenta eventos de violencia organizada en el mundo, incluyendo detalles temporales, geográficos y del tipo de conflicto. Este proyecto busca explorar patrones de violencia, tendencias temporales, distribución geográfica y posibles indicadores predictivos.

## Trabajo Práctico: Aprendizaje de Máquina

El notebook `Grupo8-UCDP-GED.ipynb` desarrolla un problema de **clasificación supervisada multiclase** para predecir la severidad de eventos de violencia organizada. La variable objetivo `severity` se construye a partir de `best` (estimación central de bajas) mediante cortes en los percentiles 33 y 66:

| Clase | Criterio | Etiqueta |
|---|---|---|
| 0 | `best` < percentil 33 | Bajo |
| 1 | percentil 33 ≤ `best` < percentil 66 | Medio |
| 2 | `best` ≥ percentil 66 | Alto |

### Análisis exploratorio

Antes del modelado se realizó un análisis descriptivo del dataset, que incluyó:

- distribución de eventos por tipo de violencia, país, región, conflicto y actores;
- evolución temporal de eventos y muertes estimadas;
- composición de las bajas entre combatientes, civiles y casos desconocidos;
- precisión geográfica de los registros;
- análisis de valores faltantes y clasificación de sus mecanismos como MAR o MNAR;
- detección de outliers mediante IQR y visualización de distribuciones en escala logarítmica;
- análisis de correlación de Spearman entre las variables de bajas.

Los outliers de `best` y `deaths_*` se conservaron en el análisis descriptivo porque representan eventos reales relevantes. Para el modelado, esas variables se excluyeron para evitar *data leakage*, ya que contienen directamente la información utilizada para construir el target.

### Preparación de los datos

El pipeline de modelado utiliza un split estratificado de 80% para entrenamiento y 20% para test. Las transformaciones que aprenden parámetros se ajustan únicamente sobre entrenamiento:

- imputación por mediana para variables numéricas y por moda para categóricas;
- extracción de año y mes a partir de las fechas;
- `civilian_ratio` como proporción de víctimas civiles;
- *frequency encoding* de la díada de actores (`dyad_freq`);
- *target encoding* suavizado de país (`country_te`);
- codificación one-hot de región, tipo de violencia y eras temporales;
- codificación cíclica del mes mediante seno y coseno;
- `RobustScaler` para variables sesgadas y `StandardScaler` para el resto.

También se analizó el balance de clases. Aunque el target se definió por percentiles, los empates en valores bajos de `best` producen un desbalance leve, con una relación aproximada de 1,5 entre la clase mayoritaria y la minoritaria. Se utilizó `class_weight='balanced'` en los modelos y se mostró `SMOTENC` como alternativa aplicada únicamente sobre entrenamiento.

### Selección y reducción de variables

Se aplicaron dos estrategias complementarias:

- **Selección por filtros:** eliminación de variables con varianza baja y selección mediante ANOVA F-score, conservando las features numéricas con relación estadísticamente significativa con `severity`.
- **PCA:** reducción de 22 features seleccionadas a 8 componentes, reteniendo 84,4% de la varianza. El análisis de loadings mostró componentes asociados principalmente con la calidad geográfica y descriptiva del registro, la dimensión temporal, la frecuencia de las díadas y la proporción de víctimas civiles.

La selección por filtros se utilizó para el modelado final porque mantiene la interpretabilidad de variables como `country_te`, `latitude`, `longitude` y `civilian_ratio`. PCA se empleó como herramienta de compresión y visualización; la proyección PC1-PC2 mostró una superposición considerable entre las clases.

### Modelos entrenados y estudiados

1. **Baseline heurístico:** predicción constante de la clase mayoritaria, utilizado como piso de comparación.
2. **Árbol de Decisión:** versión por defecto y versión optimizada con `GridSearchCV`, validación cruzada estratificada de 5 folds y optimización de Macro F1.
3. **Random Forest:** versión por defecto y versión optimizada con `RandomizedSearchCV`, combinando bootstrap y selección aleatoria de variables para reducir la varianza del árbol individual.
4. **XGBoost:** versión por defecto, análisis de *early stopping* y versión optimizada mediante `RandomizedSearchCV` sobre profundidad, tasa de aprendizaje, cantidad de árboles, muestreo y regularización.

La evaluación incluyó Accuracy, Macro F1, reportes por clase, matrices de confusión y curvas ROC One-vs-Rest. La métrica principal fue Macro F1, porque pondera por igual las tres clases.

### Resultados finales

| Modelo | Accuracy train | Accuracy test | Macro F1 train | Macro F1 test |
|---|---:|---:|---:|---:|
| Baseline heurístico | — | 0,4487 | — | 0,2065 |
| Árbol de Decisión optimizado | 0,6011 | 0,5676 | 0,5844 | 0,5488 |
| Random Forest optimizado | 0,6622 | **0,5923** | 0,6444 | **0,5661** |
| XGBoost por defecto | 0,5686 | 0,5573 | 0,4701 | 0,4566 |
| XGBoost optimizado | 0,5913 | 0,5665 | 0,5075 | 0,4781 |

El **Random Forest optimizado** obtuvo el mejor desempeño general. Todos los modelos supervisados superaron al baseline, lo que indica que las variables contextuales contienen información predictiva sobre la severidad. La clase `Medio` fue consistentemente la más difícil de identificar, debido a la superposición entre clases y a los límites definidos por percentiles sobre una variable con muchos empates.

Como limitaciones y líneas futuras se identificaron la necesidad de revisar la definición del target, incorporar contexto histórico por país y díada, ampliar la búsqueda de hiperparámetros de XGBoost, probar modelos como LightGBM y calibrar las probabilidades del modelo final.

## Estructura del repositorio

- `dataset/` - datos originales del proyecto
- `Grupo8-UCDP-GED.ipynb` - análisis exploratorio, preparación de datos y modelos de aprendizaje automático
- `README.md` - guía del repositorio
- `pyproject.toml` - dependencias y configuración de Python

## Configuración del entorno con `uv`

`uv` es una herramienta de gestión de entornos para Python que crea y sincroniza un entorno virtual basado en el archivo `pyproject.toml`.

1. Asegúrate de tener Python 3.11 o 3.12 instalado.
2. Actualiza pip:

Windows (PowerShell o CMD):

```powershell
python -m pip install --upgrade pip
```

Linux:

```bash
python3 -m pip install --upgrade pip
```

3. Instala `uv`:

Windows:

```powershell
python -m pip install uv
```

Linux:

```bash
python3 -m pip install uv
```

4. Valida que `uv` esté instalado correctamente:

```bash
uv --version
```

5. Sincroniza y crea el entorno con `uv sync` antes de instalar dependencias:

```bash
uv sync
```

6. Instala las dependencias definidas en `pyproject.toml`:

```bash
uv install
```

7. Activa el entorno:

- PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

- CMD:

```cmd
.\.venv\Scripts\activate.bat
```

- Linux / macOS:

```bash
source .venv/bin/activate
```

> Si `uv install` no funciona, puedes instalar las dependencias manualmente con `python -m pip install -r requirements.txt` si generas un `requirements.txt` o usando `pip` directamente.

## Cómo trabajar con el dataset

1. Carga los datos desde `dataset/GEDEvent_v25_1.csv`.
2. Revisa tipos de variables, valores nulos y registros duplicados.
3. Realiza análisis exploratorio con Pandas, visualizaciones con Matplotlib y Seaborn, y perfiles de datos con ydata-profiling.
4. Genera gráficos que muestren:
   - frecuencia de eventos por año y país
   - distribución de tipos de violencia organizada
   - evolución temporal de los principales actores
   - mapas o análisis geoespaciales básicos
5. Documenta hallazgos en notebooks, informes o presentaciones.

## Buenas prácticas

- No incluyas archivos de entorno virtual en el control de versiones.
- Mantén los datos originales sin modificar y trabaja sobre copias filtradas o derivadas.
- Usa notebooks claros con secciones bien identificadas.
- Guarda resultados y visualizaciones en carpetas de salida específicas si son parte del análisis.
