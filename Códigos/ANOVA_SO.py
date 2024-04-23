import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats


# Data
data = {
    "C#": [99220, 87422, 66763, 62582, 43979],
    "C++": [49203, 58019, 45756, 37618, 21923],
    "Go": [7566, 7937, 7798, 8392, 5948],
    "JavaScript": [188079, 213772, 179455, 151173, 85948],
    "Python": [223698, 284413, 251485, 225566, 130638]
}

df_gt = pd.DataFrame(data)

df_melt_gt = pd.melt(df_gt.reset_index(), id_vars=['index'], value_vars=['C#', 'C++', 'Python', 'Go', 'JavaScript'])
df_melt_gt.columns = ['index', 'Language', 'Score']

anova_gt = stats.f_oneway(df_gt['C#'], df_gt['C++'], df_gt['Python'], df_gt['Go'], df_gt['JavaScript'])
print(anova_gt)

normality_results_gt = {lang: stats.shapiro(df_gt[lang]) for lang in df_gt.columns}
print(normality_results_gt)

levene_gt = stats.levene(df_gt['C#'], df_gt['C++'], df_gt['Python'], df_gt['Go'], df_gt['JavaScript'])
print(levene_gt)

normality_pvalues = [normality_results_gt[lang][1] for lang in df_gt.columns]
plt.figure(figsize=(8, 5))
plt.bar(df_gt.columns, normality_pvalues, color='blue')
plt.title('Shapiro-Wilk Test P-values StackOverflow')
plt.ylabel('P-value')
plt.xlabel('Lenguajes de Programación')
plt.axhline(y=0.05, color='red', linestyle='--', label='Nivel de significacia (0.05)')
plt.legend()
plt.show()



