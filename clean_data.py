'''
Debes completar todas las tareas "POR HACER" según tu comprensión de la limpieza de datos.

Se recomienda dividir los pasos de limpieza complejos en funciones.

No te limites a "ejecutar"; comprende por qué realiza cada transformación.
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# Cargar datos
# ================================
df = pd.read_csv('dataset_uncleaned.csv')

# ================================
# Funciones que te pueden ayudar
# ================================

def clean_text_columns(df):
    """
    Elimina caracteres no deseados y limpia las columnas basadas en texto.
    """
    def clean_text(x):
        if pd.isna(x) or not isinstance(x, str):
            return x
        return str(x).strip('_ ,"')
    
    return df.applymap(clean_text).replace(['', 'nan', '!@9#%8', '#F%$D@*&8'], np.NaN)


def convert_credit_history_age(x):
    """
    Convertir 'X Anios Y Meses' a total de meses.
    """
    if pd.notnull(x):
        parts = x.split(' ')
        try:
            return int(parts[0]) * 12 + int(parts[3])
        except:
            return np.nan
    return x

# Agrega más funciones que te puedan ayudar si es necesario:
# - Conversion de ID/Customer_ID
# - Normalizacion de strings (e.x., para 'Type_of_Loan')
# - Limpieza de SSN
# - Imputacion Categorica
# - Tratamiento de outliers, etc.

# ================================
# Aqui comienza tu pipeline de limpieza
# ================================

# Paso 1: Limpiar datos de texto
df = clean_text_columns(df)

# Paso 2: Limpiar y convertir columnas específicas
# POR HACER: Aplicar funciones para corregir y convertir todas las columnas que lo necesiten
df['Credit_History_Age'] = df['Credit_History_Age'].apply(convert_credit_history_age).astype(float)

# POR HACER: Convertir identificadores hexadecimales a números enteros
# df['ID'] = ...
# df['Customer_ID'] = ...

# POR HACER: Corregir el formato de 'Month' a numérico (por ejemplo, January → 1)
# df['Month'] = ...

# POR HACER: Arreglar el formato 'Type_of_Loan' (eliminar 'and', hacer texto en minúsculas, etc.)
# df['Type_of_Loan'] = ...

# POR HACER: Convertir las columnas apropiadas a tipos numéricos
# e.g., df['Age'] = df['Age'].astype(int)

# POR HACER: Eliminar o corregir SSN extraños
# df['SSN'] = ...

# ================================
# Manejo de datos faltantes
# ================================

# POR HACER: Completar o imputar valores faltantes usando lógica de grupos
# p. ej., complete el nombre, la ocupación, etc. faltantes mediante el grupo del Customer_ID

# POR HACER: Detectar y corregir valores numéricos no válidos por grupo (mín./máx. o moda)
# Usar funciones para reemplazar valores fuera de rango o atípicos

# ================================
# Pasos finales
# ================================

# POR HACER: Interpolar o suavizar la edad del historial crediticio si es necesario
# df['Credit_History_Age'] = ...

# POR HACER: Cualquier conversión final que consideres necesaria hacer

# ================================
# Guardar tu dataset limpio 
# ================================
df.to_csv('dataset_cleaned.csv', index=False)
