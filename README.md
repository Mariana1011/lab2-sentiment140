# Laboratorio 2 - Sentiment140

## Descripción

En este laboratorio se desarrolló un sistema de clasificación de sentimiento utilizando el dataset Sentiment140.

El proyecto incluye experimentación reproducible, seguimiento de modelos con MLflow, trazabilidad de las contribuciones de cada integrante, registro del modelo final y una API desarrollada con FastAPI.

Las principales tecnologías utilizadas fueron:

- Python
- MLflow
- FastAPI
- Amazon EC2
- Amazon SageMaker
- scikit-learn
- Hugging Face Datasets

---

## Dataset

Dataset utilizado:

`adilbekovich/Sentiment140Twitter`

Revisión:

`b6037e127257d95b9b23d31f78b264b9ebe697fd`

El conjunto original contiene:

- Train: 1.360.000 registros
- Test: 240.000 registros

Para los experimentos se utilizó una muestra oficial de 200.000 registros.

El protocolo experimental fue:

- Muestreo: estratificado
- Tamaño de muestra: 200.000
- Random seed: 42
- Validación: StratifiedKFold
- Número de folds: 3
- Shuffle: True

El run de protocolo en MLflow es:

`450b156231234af09a6c74fb2476840a`

---

## Experimentación

Durante el laboratorio se probaron diferentes decisiones de:

### Preprocesamiento

- Eliminación de stopwords
- Conservación de negaciones
- Lematización
- Normalización de caracteres repetidos
- Tratamiento de emojis

### Representación

- Bag of Words
- TF-IDF con unigramas
- TF-IDF con unigramas y bigramas
- Vectores de spaCy

### Clasificación

- Logistic Regression
- Linear SVM
- SGDClassifier

También se realizaron experimentos de ablación para analizar el efecto de algunas decisiones sobre el rendimiento del modelo.

---

## Modelo seleccionado

La configuración seleccionada fue:

`CFG_C_LOGREG`

Run experimental seleccionado:

`205fd58983fe424f8e42b139d37fb3b2`

La configuración final utiliza:

- Conversión a minúsculas
- Sustitución de URLs
- Sustitución de menciones
- Normalización de espacios
- Normalización de caracteres repetidos
- TF-IDF con unigramas y bigramas
- Logistic Regression

---

## Modelo final

El modelo final fue entrenado utilizando los 1.360.000 registros del conjunto de entrenamiento.

Run FINAL:

`18c2076f40a44b60b7ea0a2eab001f9a`

Modelo registrado en MLflow:

`sentiment140`

Alias:

`champion`

Versión:

`1`

Resultados sobre el conjunto de test:

- Macro-F1: 0.8255561963967392
- Accuracy: 0.8255583333333333
- Macro Precision: 0.8255839035477541
- Macro Recall: 0.8255630498746328

---

## Contribuciones

### A01 - Mariana

Participó en las etapas:

- reference
- baseline
- preprocessing
- representation
- classifier
- ablation

Realizó más de tres configuraciones experimentales válidas y cumple el requisito de contribución individual.

### A02 - Sergio

Notebook SageMaker:

`nlp-lab2-sergio-v1`

Configuraciones realizadas:

- P_STOPWORDS
- P_ELONGATION
- R_TFIDF_UNI

Etapas:

- preprocessing
- representation

Cuenta con tres configuraciones válidas y dos etapas diferentes.

### A03 - César

Notebook SageMaker:

`nlp-lab2-cesar`

Configuraciones realizadas:

- P_STOPWORDS_NEGATION
- R_TFIDF_UNI_BI
- C_LINEAR_SVM

Etapas:

- preprocessing
- representation
- classifier

Cuenta con tres configuraciones válidas y tres etapas diferentes.

---

## API

La aplicación utiliza FastAPI y expone los siguientes endpoints:

- `GET /health`
- `POST /api/v1/predict`
- `GET /audit/protocol`
- `GET /audit/runs`
- `GET /audit/contributions`
- `GET /audit/model`

---

## Ejemplo de predicción

Para enviar una frase al modelo se utiliza:

~~json
{
  "text": "I love this movie so much!"
}
~~

La API responde:

~~json
{
  "model_run_id": "18c2076f40a44b60b7ea0a2eab001f9a",
  "predictions": [
    "positive"
  ]
}
~~

En este ejemplo, el modelo clasifica la frase como sentimiento positivo.

El campo `model_run_id` permite identificar exactamente qué run de MLflow realizó la predicción.

---

## Health Check

El endpoint:

`GET /health`

permite verificar que el modelo se encuentra disponible.

Respuesta esperada:

~~json
{
  "status": "ok",
  "model_run_id": "18c2076f40a44b60b7ea0a2eab001f9a"
}
~~

---

## Auditoría y trazabilidad

La API permite consultar información directamente desde MLflow.

### Protocolo

`GET /audit/protocol`

Permite revisar:

- dataset
- revisión
- estrategia de muestreo
- random seed
- estrategia de validación
- número de folds
- artifacts del protocolo

### Runs

`GET /audit/runs`

Permite consultar los runs presentados junto con:

- parámetros
- métricas
- tags
- artifacts
- configuración utilizada

### Contribuciones

`GET /audit/contributions`

Permite verificar las contribuciones de cada integrante.

Cada integrante debe contar con:

- mínimo tres configuraciones válidas
- mínimo dos etapas diferentes

Actualmente:

- A01 cumple
- A02 cumple
- A03 cumple

Además:

`invalid_run_ids = []`

`unattributed_run_ids = []`

### Modelo

`GET /audit/model`

Permite verificar la relación entre:

modelo champion → run FINAL → experimento seleccionado → protocolo

---

## Análisis de errores

El análisis de errores del modelo final se encuentra en:

`reports/error_analysis.csv`

y

`reports/error_analysis.md`

El archivo CSV contiene ejemplos de errores de clasificación y sus categorías.

El archivo Markdown presenta una interpretación de los principales patrones encontrados.

---

## Estructura del repositorio

    lab2-entrega/
    |
    |-- app/
    |   `-- main.py
    |
    |-- notebooks/
    |   `-- experiment_audit.ipynb
    |
    |-- reports/
    |   |-- error_analysis.csv
    |   `-- error_analysis.md
    |
    |-- README.md
    |-- requirements.txt
    `-- .gitignore

---

## Ejecución de la API

Instalar dependencias:

    pip install -r requirements.txt

Ejecutar FastAPI:

    uvicorn app.main:app --host 0.0.0.0 --port 8000

La API necesita acceso al servidor MLflow utilizado durante el laboratorio.

---

## Infraestructura

El laboratorio utiliza una instancia EC2 para ejecutar:

- MLflow Tracking Server
- MLflow Model Registry
- FastAPI

Los servicios están configurados para iniciar automáticamente con la instancia EC2.

Debido al entorno de AWS Academy, la dirección IPv4 pública puede cambiar cuando la instancia es detenida y encendida nuevamente.

---

## Uso de herramientas de Inteligencia Artificial

Durante el desarrollo del laboratorio se utilizó ChatGPT de OpenAI como herramienta de apoyo.

Se utilizó principalmente para:

- apoyo en la depuración de errores
- organización de comandos
- revisión de la estructura de la API
- revisión del contrato de los endpoints
- apoyo en la documentación
- explicación de errores durante la configuración de AWS y MLflow

La ejecución de los experimentos, revisión de resultados, selección del modelo y validación de la entrega fueron realizadas por los integrantes del equipo.

---

## Estado final

El sistema cuenta con:

- protocolo experimental registrado
- experimentos reproducibles
- provenance de SageMaker
- contribuciones verificables para A01, A02 y A03
- modelo final registrado
- alias `champion`
- análisis de errores
- API pública
- endpoints de auditoría
- trazabilidad entre predicción y experimentación
