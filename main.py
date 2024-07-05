import CargarDatos
import pandas as pd
import numpy as np
from scipy.stats import t
from scipy.stats import ttest_ind
from scipy.stats import f_oneway
from scipy.stats import norm
import matplotlib.pyplot as plt

a = CargarDatos.cargarDB("titanik.csv")

## Parte 1

# Calcula la edad media por género
edad_media_por_genero = a.groupby('gender')['age'].mean()

# Rellena los valores faltantes de edad con la media correspondiente al género
a['age'] = a.apply(lambda row: edad_media_por_genero[row['gender']] if pd.isnull(row['age']) else row['age'], axis=1)

# Calcula la media de las edades
media_edades = a['age'].mean()

# Calcula la mediana de las edades
mediana_edades = a['age'].median()

# Calcula la moda de las edades
moda_edades = a['age'].mode()

# Calcula el rango de las edades
rango_edades = a['age'].max() - a['age'].min()

# Calcula la varianza de las edades
varianza_edades = a['age'].var()

# Calcula la desviación estándar de las edades
desviacion_estandar_edades = a['age'].std()

# Calcula la tasa de supervivencia general
tasa_supervivencia_general = a['survived'].mean()

# Calcula la tasa de supervivencia por género
tasa_supervivencia_por_genero = a.groupby("gender")["survived"].mean()

# Realizar histogramas de las edades de los pasajeros por clase en gráficos separados
colores = ['red', 'green', 'blue']
for i in range(1, 4):
    plt.figure(figsize=(10, 6))
    plt.hist(a[a['p_class'] == i]['age'], bins=20, alpha=0.7, color=colores[i - 1], label=f'Clase {i}')
    plt.xlabel('Edad')
    plt.ylabel('Frecuencia')
    plt.title(f'Histograma de Edades de los Pasajeros de la Clase {i}')
    plt.legend()
    plt.show()

# Proponer un modelo para la distribución de la variable edad en el barco
# Dado que la distribución de la variable edad puede variar dependiendo del conjunto de datos,
# no hay un modelo único que se ajuste a todos los casos. Sin embargo, una distribución comúnmente
# utilizada para modelar la variable edad es la distribución normal (Gaussiana). Puede utilizar
# la función de densidad de probabilidad (PDF) de la distribución normal para ajustar los datos
# y estimar los parámetros de la distribución (media y desviación estándar).
# Por ejemplo:

mu, sigma = norm.fit(a['age'])
print(f"Media: {mu}, Desviación Estándar: {sigma}")

# Diagrama de cajas para las edades de los supervivientes
plt.figure(figsize=(8, 6))
plt.boxplot(a[a['survived'] == 1]['age'], vert=False)
plt.xlabel('Edad')
plt.ylabel('Supervivientes')
plt.title('Diagrama de Cajas - Edades de los supervivientes')
plt.show()

# Diagrama de cajas para las edades de los no supervivientes
plt.figure(figsize=(8, 6))
plt.boxplot(a[a['survived'] == 0]['age'], vert=False)
plt.xlabel('Edad')
plt.ylabel('No Supervivientes')
plt.title('Diagrama de Cajas - Edades de los no supervivientes')
plt.show()

## Parte 2

# Calcula el tamaño de la muestra
n = len(a['age'])

# Calcula la media de las edades
media_edades = a['age'].mean()

# Calcula la desviación estándar de las edades
desviacion_estandar_edades = a['age'].std()

# Calcula el error estándar de la media
error_estandar_media = desviacion_estandar_edades / np.sqrt(n)

# Calcula el valor crítico de la distribución t de Student
valor_critico = t.ppf(0.975, df=n - 1)

# Calcula el intervalo de confianza
intervalo_confianza = (
    media_edades - valor_critico * error_estandar_media, media_edades + valor_critico * error_estandar_media)

print(f"Intervalo de confianza (95%): {intervalo_confianza[0]} - {intervalo_confianza[1]}")

# Filtrar los datos de las mujeres interesadas en abordar el Titanic
mujeres_interesadas = a[(a['gender'] == 'female') & (a['age'].notnull())]

# Calcular el promedio de edad de las mujeres interesadas
promedio_edad_mujeres = mujeres_interesadas['age'].mean()

# Realizar la prueba de hipótesis para las mujeres
if promedio_edad_mujeres > 56:
    print(
        "Se puede afirmar con un 95% de confianza que el promedio de"
        " edad de las mujeres interesadas es mayor a 56 años."
    )
else:
    print(
        "No se puede afirmar con un 95% de confianza que el promedio"
        " de edad de las mujeres interesadas es mayor a 56 años."
    )

# Filtrar los datos de los hombres interesados en abordar el Titanic
hombres_interesados = a[(a['gender'] == 'male') & (a['age'].notnull())]

# Calcular el promedio de edad de los hombres interesados
promedio_edad_hombres = hombres_interesados['age'].mean()

# Realizar la prueba de hipótesis para los hombres
if promedio_edad_hombres > 56:
    print(
        "Se puede afirmar con un 95% de confianza que el promedio"
        " de edad de los hombres interesados es mayor a 56 años."
    )
else:
    print(
        "No se puede afirmar con un 95% de confianza que el promedio"
        " de edad de los hombres interesados es mayor a 56 años."
    )

# Prueba de hipótesis para la diferencia en la tasa de supervivencia entre hombres y mujeres

# Filtrar los datos de los hombres y mujeres
hombres = a[a['gender'] == 'male']
mujeres = a[a['gender'] == 'female']

# Realizar la prueba de hipótesis
t_statistic, p_value = ttest_ind(hombres['survived'], mujeres['survived'])

# Comprobar si la diferencia es significativa
if p_value < 0.01:
    print("Existe una diferencia significativa en la tasa de supervivencia entre hombres y mujeres.")
else:
    print("No existe una diferencia significativa en la tasa de supervivencia entre hombres y mujeres.")

# Prueba de hipótesis para la diferencia en la tasa de supervivencia en las distintas clases

# Filtrar los datos por clase
clase_1 = a[a['p_class'] == 1]
clase_2 = a[a['p_class'] == 2]
clase_3 = a[a['p_class'] == 3]

# Realizar la prueba de hipótesis
f_statistic, p_value = f_oneway(clase_1['survived'], clase_2['survived'], clase_3['survived'])

# Comprobar si la diferencia es significativa
if p_value < 0.01:
    print("Existe una diferencia significativa en la tasa de supervivencia en las distintas clases.")
else:
    print("No existe una diferencia significativa en la tasa de supervivencia en las distintas clases.")

# Prueba de hipótesis para la diferencia en la edad promedio entre hombres y mujeres

# Filtrar los datos de los hombres y mujeres
hombres = a[a['gender'] == 'male']
mujeres = a[a['gender'] == 'female']

# Realizar la prueba de hipótesis
t_statistic, p_value = ttest_ind(hombres['age'], mujeres['age'])

# Comprobar si la diferencia es significativa
if p_value < 0.05:
    print(
        "Se puede afirmar con un 95% de confianza que en promedio las mujeres"
        " eran más jóvenes que los hombres en el barco."
    )
else:
    print(
        "No se puede afirmar con un 95% de confianza que en promedio"
        " las mujeres eran más jóvenes que los hombres en el barco."
    )
