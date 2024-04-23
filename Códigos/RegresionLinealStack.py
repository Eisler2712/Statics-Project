from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

#Stackoverflow
sCsharp = [99220,87422,66763,62582,43979]
sCplus = [49203,58019,45756,37618,21923]
sGo	= [7566,7937,7798,8392,5948]
sJavaScript =[188079,213772,179455,151173,85948]
sPython	=[223698,284413,251485,225566,130638]

años = np.array([2019, 2020, 2021, 2022,2023]).reshape(-1, 1)

gh_datos = {
    'C#': np.array(sCsharp).reshape(-1, 1),
    'C++': np.array(sCplus).reshape(-1, 1),
    'Python': np.array(sPython).reshape(-1, 1),
    'Go': np.array(sGo).reshape(-1, 1),
    'JavaScript': np.array(sJavaScript).reshape(-1, 1),
}

modelos = {}
predicciones = {}
formulas = {}
años_futuros = np.array([2024, 2025]).reshape(-1, 1)

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

plt.title('Predicciones de Contribuciones en StackOverflow (2024-2025)')
plt.xlabel('Año')
plt.ylabel('Contribuciones en GitHub')
plt.legend()
plt.grid(True)
plt.show()

for lenguaje, formula in formulas.items():
    print(f'- {lenguaje}: y = {formula}')
