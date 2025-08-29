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

def clean_ssn(x):
    """
    Limpia el SSN para asegurarse de que esté en formato válido (XXX-XX-XXXX).
    """
    if pd.notnull(x) and isinstance(x, str):
        return x.strip()
    return np.nan

def convert_month_to_numeric(x):
    """
    Convierte el nombre del mes a un valor numérico (e.g., January -> 1).
    """
    months = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
        'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    return months.get(x, np.nan)

def clean_type_of_loan(x):
    """
    Limpia la columna 'Type_of_Loan' eliminando 'and' y pasando a minúsculas.
    """
    if pd.notnull(x) and isinstance(x, str):
        return x.replace('and', '').lower()
    return np.nan

# ================================
# Aqui comienza tu pipeline de limpieza
# ================================

# Paso 1: Limpiar datos de texto
df = clean_text_columns(df)

# Paso 2: Limpiar y convertir columnas específicas
# Aplicar funciones para corregir y convertir todas las columnas que lo necesiten
df['Credit_History_Age'] = df['Credit_History_Age'].apply(convert_credit_history_age).astype(float)

# Convertir identificadores hexadecimales a números enteros
df['ID'] = df['ID'].apply(lambda x: int(x, 16) if pd.notnull(x) else np.nan)
df['Customer_ID'] = df['Customer_ID'].apply(lambda x: int(x.split('_')[1], 16) if pd.notnull(x) else np.nan)

# Corregir el formato de 'Month' a numérico (por ejemplo, January → 1)
df['Month'] = df['Month'].apply(convert_month_to_numeric)

# Arreglar el formato 'Type_of_Loan' (eliminar 'and', hacer texto en minúsculas, etc.)
df['Type_of_Loan'] = df['Type_of_Loan'].apply(clean_type_of_loan)

# Convertir las columnas apropiadas a tipos numéricos
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Annual_Income'] = pd.to_numeric(df['Annual_Income'], errors='coerce')
df['Monthly_Inhand_Salary'] = pd.to_numeric(df['Monthly_Inhand_Salary'], errors='coerce')
df['Num_Bank_Accounts'] = pd.to_numeric(df['Num_Bank_Accounts'], errors='coerce')

# Eliminar o corregir SSN extraños
df['SSN'] = df['SSN'].apply(clean_ssn)

# ================================
# Manejo de datos faltantes
# ================================

# Completar o imputar valores faltantes usando lógica de grupos
# Aquí se asume que los valores faltantes se completan con la moda dentro de cada 'Customer_ID'
df['Name'] = df.groupby('Customer_ID')['Name'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
df['Occupation'] = df.groupby('Customer_ID')['Occupation'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))

# Detectar y corregir valores numéricos no válidos por grupo (mín./máx. o moda)
# En este ejemplo, se usa el valor promedio para los valores atípicos
df['Age'] = df.groupby('Customer_ID')['Age'].transform(lambda x: x.fillna(x.mean()))
df['Annual_Income'] = df.groupby('Customer_ID')['Annual_Income'].transform(lambda x: x.fillna(x.mean()))

# ================================
# Pasos finales (hechos conforme fui viendo el csv resultante)
# ================================

# Interpolar o suavizar la edad del historial crediticio si es necesario
df['Credit_History_Age'] = df['Credit_History_Age'].interpolate(method='linear')

# Cualquier conversión final que consideres necesaria hacer
# Aquí podrías aplicar cualquier otro ajuste o normalización final

# Reemplazar todos los valores negativos en la columna 'Age' por NaN
df['Age'] = df['Age'].apply(lambda x: np.nan if x < 0 else x)
df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].replace('NM', np.nan)

# Asignar un nombre a cada 'Customer_ID' y reemplazar los nulos en 'Name' por el nombre correspondiente
df['Name'] = df.groupby('Customer_ID')['Name'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
df['Age'] = df.groupby('Customer_ID')['Age'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
df['SSN'] = df.groupby('Customer_ID')['SSN'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
df['Monthly_Inhand_Salary'] = df.groupby('Customer_ID')['Monthly_Inhand_Salary'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
df['Credit_Mix'] = df.groupby('Customer_ID')['Credit_Mix'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
df['Payment_of_Min_Amount'] = df.groupby('Customer_ID')['Payment_of_Min_Amount'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
df['Changed_Credit_Limit'] = df.groupby('Customer_ID')['Changed_Credit_Limit'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
df['Num_Credit_Inquiries'] = df.groupby('Customer_ID')['Num_Credit_Inquiries'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))

df['Num_of_Delayed_Payment'] = df['Num_of_Delayed_Payment'].fillna(0)
df['Amount_invested_monthly'] = df['Amount_invested_monthly'].fillna(0)
df['Monthly_Balance'] = df['Monthly_Balance'].fillna(0)

df['Num_Bank_Accounts'] = pd.to_numeric(df['Num_Bank_Accounts'], errors='coerce')
df['Num_Credit_Card'] = pd.to_numeric(df['Num_Credit_Card'], errors='coerce')
df['Interest_Rate'] = pd.to_numeric(df['Interest_Rate'], errors='coerce')
df['Num_of_Loan'] = pd.to_numeric(df['Num_of_Loan'], errors='coerce')
df['Changed_Credit_Limit'] = pd.to_numeric(df['Changed_Credit_Limit'], errors='coerce')
df['Num_Credit_Inquiries'] = pd.to_numeric(df['Num_Credit_Inquiries'], errors='coerce')

lower_bound = {
    'Num_Bank_Accounts': df['Num_Bank_Accounts'].quantile(0.05),
    'Num_Credit_Card': df['Num_Credit_Card'].quantile(0.05),
    'Interest_Rate': df['Interest_Rate'].quantile(0.05),
    'Num_of_Loan': df['Num_of_Loan'].quantile(0.05),
    'Changed_Credit_Limit': df['Changed_Credit_Limit'].quantile(0.05),
    'Num_Credit_Inquiries': df['Num_Credit_Inquiries'].quantile(0.05),
    'Annual_Income': df['Annual_Income'].quantile(0.05)
}

upper_bound = {
    'Num_Bank_Accounts': df['Num_Bank_Accounts'].quantile(0.95),
    'Num_Credit_Card': df['Num_Credit_Card'].quantile(0.95),
    'Interest_Rate': df['Interest_Rate'].quantile(0.95),
    'Num_of_Loan': df['Num_of_Loan'].quantile(0.95),
    'Changed_Credit_Limit': df['Changed_Credit_Limit'].quantile(0.95),
    'Num_Credit_Inquiries': df['Num_Credit_Inquiries'].quantile(0.95),
    'Annual_Income': df['Annual_Income'].quantile(0.95)
}

# Reemplazar valores atípicos con la moda de cada grupo 'Customer_ID'
df['Num_Bank_Accounts'] = df.groupby('Customer_ID')['Num_Bank_Accounts'].transform(
    lambda x: x.apply(lambda v: x.mode()[0] if v < lower_bound['Num_Bank_Accounts'] or v > upper_bound['Num_Bank_Accounts'] else v))
df['Num_Credit_Card'] = df.groupby('Customer_ID')['Num_Credit_Card'].transform(
    lambda x: x.apply(lambda v: x.mode()[0] if v < lower_bound['Num_Credit_Card'] or v > upper_bound['Num_Credit_Card'] else v))
df['Interest_Rate'] = df.groupby('Customer_ID')['Interest_Rate'].transform(
    lambda x: x.apply(lambda v: x.mode()[0] if v < lower_bound['Interest_Rate'] or v > upper_bound['Interest_Rate'] else v))
df['Num_of_Loan'] = df.groupby('Customer_ID')['Num_of_Loan'].transform(
    lambda x: x.apply(lambda v: x.mode()[0] if v < lower_bound['Num_of_Loan'] or v > upper_bound['Num_of_Loan'] else v))
df['Changed_Credit_Limit'] = df.groupby('Customer_ID')['Changed_Credit_Limit'].transform(
    lambda x: x.apply(lambda v: x.mode()[0] if v < lower_bound['Changed_Credit_Limit'] or v > upper_bound['Changed_Credit_Limit'] else v))
df['Num_Credit_Inquiries'] = df.groupby('Customer_ID')['Num_Credit_Inquiries'].transform(
    lambda x: x.apply(lambda v: x.mode()[0] if v < lower_bound['Num_Credit_Inquiries'] or v > upper_bound['Num_Credit_Inquiries'] else v))
df['Annual_Income'] = df.groupby('Customer_ID')['Annual_Income'].transform(
    lambda x: x.apply(lambda v: x.mode()[0] if v < lower_bound['Annual_Income'] or v > upper_bound['Annual_Income'] else v))



null_counts = df.isnull().sum()
print(null_counts)

# ================================
# Guardar tu dataset limpio 
# ================================
df.to_csv('dataset_cleaned.csv', index=False)
# Delay_from_due_date no supe calcularlo, ni Payment_Behaviour. Son datos un poco más difíciles de sacar para rellenado
# los errores que tenga este código van a ser por falta de comprensión del dataset
