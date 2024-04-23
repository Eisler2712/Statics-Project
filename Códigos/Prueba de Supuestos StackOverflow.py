from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox

sCsharp = np.array([99220,87422,66763,62582,43979])
sCplus = np.array([49203,58019,45756,37618,21923])
sGo = np.array([7566,7937,7798,8392,5948])
sJavaScript = np.array([188079,213772,179455,151173,85948])
sPython = np.array([223698,284413,251485,225566,130638])

scaler = MinMaxScaler()

años = np.array([2019, 2020, 2021, 2022, 2023]).reshape(-1, 1)

datoss = {
    'C#': scaler.fit_transform(np.vstack([sCsharp]).T),
    'C++': scaler.fit_transform(np.vstack([sCplus]).T),
    'Python': scaler.fit_transform(np.vstack([sPython]).T),
    'Go': scaler.fit_transform(np.vstack([sGo]).T),
    'JavaScript': scaler.fit_transform(np.vstack([sJavaScript]).T)
}

modelos = {}
residuos = {}
for lenguaje, datos in datoss.items():
    modelo = LinearRegression().fit(años, datos)
    modelos[lenguaje] = modelo
    residuos[lenguaje] = datos - modelo.predict(años)

resultados_ljungbox = {}
for lenguaje, residuo in residuos.items():
    resultados_ljungbox[lenguaje] = acorr_ljungbox(residuo.flatten(), lags=1, return_df=True)

plt.figure(figsize=(10, 6))

lenguajes = list(resultados_ljungbox.keys())

lb_stats = [resultados_ljungbox[lang]['lb_stat'].iloc[0] for lang in lenguajes]
lb_pvalues = [resultados_ljungbox[lang]['lb_pvalue'].iloc[0] for lang in lenguajes]

x_pos = np.arange(len(lenguajes))

plt.bar(x_pos, lb_stats, width=0.4, label='Estadístico Ljung-Box', alpha=0.6)

ax2 = plt.twinx()
ax2.plot(x_pos, lb_pvalues, color='red', label='Valor p', marker='o', linewidth=2)

plt.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.xticks(x_pos, lenguajes)
plt.xlabel('Lenguaje de Programación')
plt.ylabel('Estadístico Ljung-Box')
ax2.set_ylabel('Valor p')

plt.title('Resultados de la Prueba de Autocorrelación Ljung-Box por Lenguaje de Programación')
plt.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')
plt.show()

plt.figure(figsize=(15, 10))

for i, (lenguaje, datos) in enumerate(datoss.items(), 1):
    modelo = modelos[lenguaje]
    predicciones = modelo.predict(años)
    residuales = datos.flatten() - predicciones.flatten()

    plt.subplot(2, 3, i)
    plt.scatter(predicciones, residuales)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'Homocedasticidad de {lenguaje}')
    plt.xlabel('Predicciones')
    plt.ylabel('Residuos')

plt.tight_layout()
plt.show()

# Gráficos Q-Q para la normalidad de los residuos
plt.figure(figsize=(15, 10))

for i, (lenguaje, datos) in enumerate(datoss.items(), 1):
    modelo = modelos[lenguaje]
    predicciones = modelo.predict(años)
    residuales = datos.flatten() - predicciones.flatten()

    # Gráfico Q-Q para normalidad
    plt.subplot(2, 3, i)
    stats.probplot(residuales, dist="norm", plot=plt)
    plt.title(f'Gráfico Q-Q de {lenguaje}')

plt.tight_layout()
plt.show()




