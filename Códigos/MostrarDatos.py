from sklearn.preprocessing import MinMaxScaler
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

todos_los_datos_gt = np.array([gtCsharp, gtCplus, gtPython, gtGo, gtJavaScript]).flatten().reshape(-1, 1)
todos_los_datos_so = np.array([sCsharp, sCplus, sPython, sGo, sJavaScript]).flatten().reshape(-1, 1)
todos_los_datos_gh = np.array([ghCsharp, ghCplus, ghPython, ghGo, ghJavaScript]).flatten().reshape(-1, 1)

datos_normalizados_gt = scaler.fit_transform(todos_los_datos_gt).flatten()
datos_normalizados_so = scaler.fit_transform(todos_los_datos_so).flatten()
datos_normalizados_gh = scaler.fit_transform(todos_los_datos_gh).flatten()

lenguajes = ['C#', 'C++', 'Python', 'Go', 'JavaScript']
datos_por_lenguaje_gt = np.split(datos_normalizados_gt, 5)
datos_por_lenguaje_so = np.split(datos_normalizados_so, 5)
datos_por_lenguaje_gh = np.split(datos_normalizados_gh, 5)

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
for i, lenguaje in enumerate(lenguajes):
    plt.plot(años.flatten(), datos_por_lenguaje_gt[i], label=lenguaje)
plt.title('Google Trends (Normalizado)')
plt.xlabel('Año')
plt.ylabel('Interés Normalizado')
plt.legend()

plt.subplot(1, 3, 2)
for i, lenguaje in enumerate(lenguajes):
    plt.plot(años.flatten(), datos_por_lenguaje_so[i], label=lenguaje)
plt.title('StackOverflow (Normalizado)')
plt.xlabel('Año')
plt.ylabel('Contribuciones Normalizadas')
plt.legend()


plt.subplot(1, 3, 3)
for i, lenguaje in enumerate(lenguajes):
    plt.plot(años.flatten(), datos_por_lenguaje_gh[i], label=lenguaje)
plt.title('GitHub (Normalizado)')
plt.xlabel('Año')
plt.ylabel('Contribuciones Normalizadas')
plt.legend()

plt.tight_layout()
plt.show()

