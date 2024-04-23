from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

#Google Trends
gtCsharp = [3186,3517,3004,3790,3275]
gtCplus = [2480,3047,2919,4434,3536]
gtPython =[2418,3150,2886,4445,4127]
gtGo =[2388,2891,2566,4386,4095]
gtJavaScript = [2869,3313,2808,4413,3908]

años = np.array([2019, 2020, 2021, 2022,2023]).reshape(-1, 1)

gh_datos = {
    'C#': np.array(gtCsharp).reshape(-1, 1),
    'C++': np.array(gtCplus).reshape(-1, 1),
    'Python': np.array(gtPython).reshape(-1, 1),
    'Go': np.array(gtGo).reshape(-1, 1),
    'JavaScript': np.array(gtJavaScript).reshape(-1, 1),
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

plt.title('Predicciones de Contribuciones en Google Trends (2024-2025)')
plt.xlabel('Año')
plt.ylabel('Contribuciones en GitHub')
plt.legend()
plt.grid(True)
plt.show()

for lenguaje, formula in formulas.items():
    print(f'- {lenguaje}: y = {formula}')