import numpy as np
import matplotlib.pyplot as plt

languages = ["C#", "C++", "Go", "JavaScript", "Python"]

sCsharp = np.array([99220, 87422, 66763, 62582, 43979])
sCplus = np.array([49203, 58019, 45756, 37618, 21923])
sGo = np.array([7566, 7937, 7798, 8392, 5948])
sJavaScript = np.array([188079, 213772, 179455, 151173, 85948])
sPython = np.array([223698, 284413, 251485, 225566, 130638])

ghCsharp = np.array([1210, 1655, 2111, 2054, 2568])
ghCplus = np.array([1671, 1954, 2193, 2502, 3264])
ghGo = np.array([600, 673, 851, 956, 1100])
ghJavaScript = np.array([7201, 12424, 16243, 16828, 19503])
ghPython = np.array([4391, 6473, 7700, 8543, 10482])

# Cálculo de correlaciones de Pearson
correlations = {
    "C#": np.corrcoef(sCsharp, ghCsharp)[0, 1],
    "C++": np.corrcoef(sCplus, ghCplus)[0, 1],
    "Go": np.corrcoef(sGo, ghGo)[0, 1],
    "JavaScript": np.corrcoef(sJavaScript, ghJavaScript)[0, 1],
    "Python": np.corrcoef(sPython, ghPython)[0, 1]
}

fig, axes = plt.subplots(5, 1, figsize=(10, 20))

data_stack = [sCsharp, sCplus, sGo, sJavaScript, sPython]
data_github = [ghCsharp, ghCplus, ghGo, ghJavaScript, ghPython]

years = ["2019", "2020", "2021", "2022", "2023"]

for i, ax in enumerate(axes):
    ax.plot(years, data_stack[i], label=f'{languages[i]} StackOverflow', marker='o')
    ax.plot(years, data_github[i], label=f'{languages[i]} Github', marker='o')
    ax.set_title(f'{languages[i]} SO vs. GH Actividad')
    ax.set_xlabel('Años')
    ax.set_ylabel('Actividad')
    ax.legend()

plt.tight_layout()
plt.show()
