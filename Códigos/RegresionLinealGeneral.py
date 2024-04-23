from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

#Google Trends
gtCsharp = [3186,3517,3004,3790,3275]
gtCplus = [2480,3047,2919,4434,3536]
gtPython =[2418,3150,2886,4445,4127]
gtGo =[2388,2891,2566,4386,4095]
gtJavaScript = [2869,3313,2808,4413,3908]

#Stackoverflow
sCsharp = [99220,87422,66763,62582,43979]
sCplus = [49203,58019,45756,37618,21923]
sGo	= [7566,7937,7798,8392,5948]
sJavaScript =[188079,213772,179455,151173,85948]
sPython	=[223698,284413,251485,225566,130638]

#Github
ghCsharp = [1210,1655,2111,2054,2568]
ghCplus = [1671,1954,2193,2502,3264]
ghGo = [600,673,851,956,1100]
ghJavaScript = [7201,12424,16243,16828,19503]
ghPython = [4391,6473,7700,8543,10482]

scaler = MinMaxScaler()

años = np.array([2019, 2020, 2021, 2022,2023]).reshape(-1, 1)
años_futuros = np.array([2024, 2025]).reshape(-1, 1)

datos_combinados = {
    lenguaje: scaler.fit_transform(np.vstack([gt, s, gh]).T)
    for lenguaje, gt, s, gh in zip(['C#', 'C++', 'Python', 'Go', 'JavaScript'],
                                   [gtCsharp, gtCplus, gtPython, gtGo, gtJavaScript],
                                   [sCsharp, sCplus, sPython, sGo, sJavaScript],
                                   [ghCsharp, ghCplus, ghPython, ghGo, ghJavaScript])
}

metrica_compuesta = {lenguaje: np.mean(datos, axis=1) for lenguaje, datos in datos_combinados.items()}

modelos_compuestos = {}
predicciones_compuestas = {}
formulas = {}

for lenguaje, metricas in metrica_compuesta.items():
    modelo = LinearRegression().fit(años, metricas.reshape(-1, 1))
    modelos_compuestos[lenguaje] = modelo
    predicciones_compuestas[lenguaje] = modelo.predict(años_futuros).flatten()
    m = modelo.coef_[0][0]
    b = modelo.intercept_[0]
    formulas[lenguaje] = f'{m:.2f}x + {b:.2f}'

plt.figure(figsize=(10, 6))

años_total_compuestos = np.concatenate((años, años_futuros))

for lenguaje, metricas in metrica_compuesta.items():
    plt.scatter(años, metricas, label=f'{lenguaje} Actual')

    predicciones_extendidas_compuestas = modelos_compuestos[lenguaje].predict(
        años_total_compuestos.reshape(-1, 1)).flatten()
    plt.plot(años_total_compuestos, predicciones_extendidas_compuestas, '--', label=f'{lenguaje} Predicción')

plt.title('Predicciones Compuestas (2024-2025)')
plt.xlabel('Año')
plt.ylabel('Métrica Compuesta Normalizada')
plt.legend()
plt.grid(True)
plt.show()

for lenguaje, formula in formulas.items():
    print(f'- {lenguaje}: y = {formula}')
