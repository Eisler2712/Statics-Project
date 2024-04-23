import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# Data
data = {
    "C#": [3186, 3517, 3004, 3790, 3275],
    "C++": [2480, 3047, 2919, 4434, 3536],
    "Python": [2418, 3150, 2886, 4445, 4127],
    "Go": [2388, 2891, 2566, 4386, 4095],
    "JavaScript": [2869, 3313, 2808, 4413, 3908]
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
plt.title('Shapiro-Wilk Test P-values GoogleTrends')
plt.ylabel('P-value')
plt.xlabel('Lenguajes de Programación')
plt.axhline(y=0.05, color='red', linestyle='--', label='Nivel de significacia (0.05)')
plt.legend()
plt.show()





