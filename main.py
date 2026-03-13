import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Antonio Eugenio daniel

# 1. Cargar el archivo CSV
df = pd.read_csv("penguins.csv")
df = df.dropna() # Limpieza de datos nulos

# 2. Preparar variables (X: características, y: especie real)
X = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
y = df['species']

# Dividir en entrenamiento (train) y prueba (test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Función del Clasificador Humano
def clasificador_humano(bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g):
    if flipper_length_mm > 206:
        return "Gentoo"
    
    if bill_length_mm > 43.3:
        return "Chinstrap"
    
    else:
        return "Adelie"

# Realizar predicciones humanas
predicciones_humano = [clasificador_humano(r['bill_length_mm'], r['bill_depth_mm'], 
                                         r['flipper_length_mm'], r['body_mass_g']) 
                       for _, r in X_test.iterrows()]

# 4. Entrenar Clasificador ML
modelo_ml = DecisionTreeClassifier(random_state=42)
modelo_ml.fit(X_train, y_train)

# Realizar predicciones ML
predicciones_ml = modelo_ml.predict(X_test)

# 5. Guardar resultados en un nuevo CSV
df_resultado = pd.DataFrame({
    'prediccion_humano': predicciones_humano,
    'prediccion_ml': predicciones_ml
})

df_resultado.to_csv("predicciones_finales.csv", index=False)
print("Archivo generado exitosamente.")