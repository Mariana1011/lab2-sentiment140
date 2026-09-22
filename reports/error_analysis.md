# Análisis de errores

Se seleccionaron aleatoriamente 20 errores del modelo final utilizando
semilla 42. La muestra contiene errores correspondientes a ambas clases.

## Frecuencia por categoría

- **mixed:** 8
- **other:** 5
- **informal:** 4
- **sarcasm:** 2
- **contrast:** 1

## Patrones principales

### 1. Mixed

El patrón más frecuente corresponde a mensajes con polaridad mixta.
Varios tweets contienen simultáneamente expresiones negativas y positivas,
por ejemplo una queja seguida de agradecimiento, una experiencia negativa
con un cierre positivo o una frase de frustración acompañada de entusiasmo.
Un modelo lineal basado en TF-IDF puede dar mayor peso a algunas palabras
individuales sin representar completamente la relación entre las distintas
partes del mensaje.

### 2. Other / informal

Otro patrón frecuente corresponde a mensajes cuyo sentimiento depende del
contexto conversacional o del lenguaje informal típico de Twitter. Se
observan abreviaciones, errores ortográficos, expresiones coloquiales y
mensajes cuyo significado completo depende de conversaciones anteriores.
Estos fenómenos reducen la información disponible para un clasificador
basado principalmente en características léxicas.

También aparecen casos de sarcasmo, donde palabras aparentemente positivas,
como "thanks" o "interesting", expresan realmente una intención negativa.

## Conclusión

Los errores sugieren que el modelo funciona bien ante señales léxicas
claras, pero presenta mayor dificultad cuando la polaridad depende de
contraste, contexto, lenguaje informal o intención pragmática como el
sarcasmo.
