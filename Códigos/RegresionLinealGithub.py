from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

#Github
ghCsharp = [1210,1655,2111,2054,2568]
ghCplus = [1671,1954,2193,2502,3264]
ghGo = [600,673,851,956,1100]
ghJavaScript = [7201,12424,16243,16828,19503]
ghPython = [4391,6473,7700,8543,10482]

años = np.array([2019, 2020, 2021, 2022,2023]).reshape(-1, 1)

gh_datos = {
    'C#': np.array(ghCsharp).reshape(-1, 1),
    'C++': np.array(ghCplus).reshape(-1, 1),
    'Python': np.array(ghPython).reshape(-1, 1),
    'Go': np.array(ghGo).reshape(-1, 1),
    'JavaScript': np.array(ghJavaScript).reshape(-1, 1),
}

modelos = {}
predicciones = {}
años_futuros = np.array([2024, 2025]).reshape(-1, 1)
formulas = {}

for lenguaje, datos in gh_datos.items():
    modelo = LinearRegression().fit(años, datos)
    modelos[lenguaje] = modelo
    predicciones[lenguaje] = modelo.predict(años_futuros).flatten()
    m = modelo.coef_[0][0]
    b = modelo.intercept_[0]
    formulas[lenguaje] = f'{m:.2f}x + {b:.2f}'

plt.figure(figsize=(12, 8))

años_total = np.concatenate((años, años_futuros))

for lenguaje, modelo in modelos.items():
    plt.scatter(años, gh_datos[lenguaje], label=f'{lenguaje} Actual')
    predicciones_extendidas = modelo.predict(años_total.reshape(-1, 1)).flatten()
    plt.plot(años_total, predicciones_extendidas, '--', label=f'{lenguaje} Predicción')

plt.title('Predicciones de Contribuciones en GitHub (2024-2025)')
plt.xlabel('Año')
plt.ylabel('Contribuciones en GitHub')
plt.legend()
plt.grid(True)
plt.show()
#mostrar formulas
for lenguaje, formula in formulas.items():
    print(f'{lenguaje}: {formula}')