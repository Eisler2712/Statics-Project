from scipy.stats import chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Datos iniciales
gtCsharp = [3186, 3517, 3004, 3790, 3275]
gtCplus = [2480, 3047, 2919, 4434, 3536]
gtPython =[2418, 3150, 2886, 4445, 4127]
gtGo =[2388, 2891, 2566, 4386, 4095]
gtJavaScript = [2869, 3313, 2808, 4413, 3908]

sCsharp = [99220, 87422, 66763, 62582, 43979]
sCplus = [49203, 58019, 45756, 37618, 21923]
sGo = [7566, 7937, 7798, 8392, 5948]
sJavaScript =[188079, 213772, 179455, 151173, 85948]
sPython =[223698, 284413, 251485, 225566, 130638]

ghCsharp = [1210, 1655, 2111, 2054, 2568]
ghCplus = [1671, 1954, 2193, 2502, 3264]
ghGo = [600, 673, 851, 956, 1100]
ghJavaScript = [7201, 12424, 16243, 16828, 19503]
ghPython = [4391, 6473, 7700, 8543, 10482]

def calculate_relative_growth(initial, final):
    return (final - initial) / initial

growth = {
    'Google Trends': {
        'C#': calculate_relative_growth(gtCsharp[0], gtCsharp[-1]),
        'C++': calculate_relative_growth(gtCplus[0], gtCplus[-1]),
        'Python': calculate_relative_growth(gtPython[0], gtPython[-1]),
        'Go': calculate_relative_growth(gtGo[0], gtGo[-1]),
        'JavaScript': calculate_relative_growth(gtJavaScript[0], gtJavaScript[-1]),
    },
    'Stack Overflow': {
        'C#': calculate_relative_growth(sCsharp[0], sCsharp[-1]),
        'C++': calculate_relative_growth(sCplus[0], sCplus[-1]),
        'Python': calculate_relative_growth(sPython[0], sPython[-1]),
        'Go': calculate_relative_growth(sGo[0], sGo[-1]),
        'JavaScript': calculate_relative_growth(sJavaScript[0], sJavaScript[-1]),
    },
    'GitHub': {
        'C#': calculate_relative_growth(ghCsharp[0], ghCsharp[-1]),
        'C++': calculate_relative_growth(ghCplus[0], ghCplus[-1]),
        'Python': calculate_relative_growth(ghPython[0], ghPython[-1]),
        'Go': calculate_relative_growth(ghGo[0], ghGo[-1]),
        'JavaScript': calculate_relative_growth(ghJavaScript[0], ghJavaScript[-1]),
    }
}

df_growth = pd.DataFrame(growth)

categories = df_growth.map(lambda x: 'Creciente' if x > 0.1 else ('Decreciente' if x < -0.1 else 'Estable'))

category_counts = categories.apply(pd.Series.value_counts).fillna(0).astype(int)

chi2, p, dof, expected = chi2_contingency(category_counts.T)

plt.figure(figsize=(12, 8))
sns.barplot(data=df_growth.T, palette='viridis')
plt.title('Crecimiento Relativo de Lenguajes de Programación por Plataforma')
plt.ylabel('Crecimiento Relativo')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()


category_counts.T.plot(kind='bar', figsize=(12, 8), width=0.8)
plt.title('Distribución de Categorías de Crecimiento por Plataforma')
plt.ylabel('Número de Lenguajes')
plt.xlabel('Plataforma')
plt.xticks(rotation=0)
plt.legend(title='Crecimiento')
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()

print(f"Chi-squared: {chi2}, p-value: {p}")

