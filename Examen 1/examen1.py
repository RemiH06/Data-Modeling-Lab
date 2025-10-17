#!/usr/bin/env python
# coding: utf-8

# ## Sección Conceptual

# Pregunta 1. Explica brevemente la diferencia entre overfitting y underfitting y describe un ejemplo de cada uno en el contexto del dataset de atletas.
# 
# Respuesta:
# - Overfitting ocurre cuando el modelo se ajusta demasiado a los datos de entrenamiento y no generaliza bien a los datos nuevos. Ejemplo: El modelo predice perfectamente el rendimiento en el entrenamiento, pero mal en la prueba.
# - Underfitting sucede cuando el modelo no captura correctamente la relación entre las variables. Ejemplo: Un modelo simple que no explica bien las variaciones en el rendimiento.

# Pregunta 2. ¿Qué tipo de problemas podría presentar una regresión lineal simple con este dataset? Menciona dos posibles violaciones de sus supuestos y cómo podrías detectarlas.
# 
# Respuesta:
# - Multicolinealidad: Si las variables están correlacionadas (ej. horas_entrenamiento y vo2max), puede generar estimaciones inestables. Se puede detectar con VIF.
# - Heterocedasticidad: La varianza de los errores no es constante. Se puede detectar con un gráfico de dispersión de los residuos.

# Pregunta 3. Explica en tus palabras cómo actúan las penalizaciones Ridge y Lasso, y qué efecto esperarías si las variables horas_entrenamiento y vo2max están muy correlacionadas.
# 
# Respuesta:
# - Ridge reduce los coeficientes grandes, pero no los elimina.
# - Lasso puede poner a cero los coeficientes de variables, eliminándolas.
# Si horas_entrenamiento y vo2max están correlacionadas, Ridge reducirá ambos coeficientes, mientras que Lasso podría eliminar uno.

# Pregunta 4. Menciona una situación en la que preferirías usar un modelo no lineal (como un Random Forest o SVR) sobre una regresión lineal en este contexto.
# 
# Respuesta:
# - Usaría un modelo no lineal si las relaciones entre las variables y el rendimiento no son lineales o si hay interacciones complejas entre las variables que la regresión lineal no puede capturar.

# ## Sección Práctica

# Pregunta 5: Carga el dataset y separa los datos en entrenamiento (80%) y prueba (20%). Ajusta una regresión lineal para predecir rendimiento. Reporta el RMSE en entrenamiento y prueba. ¿Observas signos de overfitting? Justifica.
# 
# Respuesta:
# - Si el RMSE de entrenamiento es mucho menor que el RMSE de prueba, es probable que el modelo esté sobreajustado. Para el tamaño del dataset lo prudente sería tener un 70-30

# Pregunta 6: Ajusta un modelo con Ridge, buscando el hiperparámetro óptimo de regularización con GridSearchCV. Indica el valor óptimo de α (alpha) encontrado. Compara su rendimiento con la regresión lineal base, ¿Qué modelo prefieres y por qué?
# 
# Respuesta:
# - El valor óptimo de alpha fue 10.0. Prefiero Ridge si tiene un mejor RMSE en comparación con la regresión lineal, ya que reduce el riesgo de overfitting.

# Pregunta 8: Ajusta un Random Forest Regressor. Usa validación cruzada (k=5). Reporta el RMSE promedio y muestra las 3 variables más importantes. ¿Qué conclusiones puedes sacar sobre la influencia de las variables más relevantes?
# 
# Respuesta:
# - El RMSE promedio con validación cruzada fue reportado. Las 3 variables más importantes fueron porcentaje_grasa, vo2max, y horas_entrenamiento, lo que sugiere que estos factores son cruciales para predecir el rendimiento.

# In[18]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


# In[19]:


# Cargar el dataset
df = pd.read_csv('rendimiento_atletas.csv')

# Cambiar 'M' por 0 y 'F' por 1 en la columna 'sexo'
df['sexo'] = df['sexo'].replace({'M': 0, 'F': 1})


# In[20]:


# Separar en características y objetivo
X = df.drop('rendimiento', axis=1)
y = df['rendimiento']

# Dividir en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Estandarizar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Ajustar un modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Hacer predicciones
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Calcular RMSE
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f'RMSE Entrenamiento: {rmse_train}')
print(f'RMSE Prueba: {rmse_test}')


# In[21]:


# Ajuste del modelo Ridge con GridSearchCV para encontrar el mejor alpha
ridge = Ridge()
param_grid = {'alpha': np.logspace(-6, 6, 13)}
grid_search = GridSearchCV(ridge, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Mejor parámetro alpha
best_alpha = grid_search.best_params_['alpha']
print(f'Alpha óptimo: {best_alpha}')


# In[22]:


# Comparar rendimiento con la regresión lineal base
ridge_model = grid_search.best_estimator_
y_pred_ridge = ridge_model.predict(X_test)

rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
print(f'RMSE Ridge: {rmse_ridge}')


# In[23]:


# Ajustar el modelo Random Forest
rf_model = RandomForestRegressor(random_state=42)

# Validación cruzada con k=5
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
mean_rmse = -cv_scores.mean()

print(f'RMSE Promedio (Random Forest): {mean_rmse}')


# In[24]:


# Mostrar las 3 variables más importantes
importances = rf_model.fit(X_train, y_train).feature_importances_
indices = np.argsort(importances)[::-1]

print("Las 3 variables más importantes son:")
for i in range(3):
    print(f'{X.columns[indices[i]]}: {importances[indices[i]]}')


# ## Sección Reflexión

# Pregunta 9: Supón que te piden implementar este modelo en un centro de alto rendimiento con recursos limitados. ¿Qué modelo elegirías y por qué? ¿Qué variables eliminarías o transformarías antes de implementarlo? Explica cómo comunicarías los resultados a un entrenador que no sabe de estadística, resaltando los factores clave que influyen en el rendimiento.
# 
# Respuesta:
# - Elegiría Ridge Regression por su simplicidad y eficiencia. Eliminaría variables redundantes y transformaría las categóricas. Explicaría al entrenador que porcentaje de grasa, VO2 max, y horas de entrenamiento son los factores clave que influyen en el rendimiento.
