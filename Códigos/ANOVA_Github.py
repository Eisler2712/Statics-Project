import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# Data
data = {
    "C#": [1210, 1655, 2111, 2054, 2568],
    "C++": [1671, 1954, 2193, 2502, 3264],
    "Go": [600, 673, 851, 956, 1100],
    "JavaScript": [7201, 12424, 16243, 16828, 19503],
    "Python": [4391, 6473, 7700, 8543, 10482]
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
plt.title('Shapiro-Wilk Test P-values Github')
plt.ylabel('P-value')
plt.xlabel('Lenguajes de Programación')
plt.axhline(y=0.05, color='red', linestyle='--', label='Nivel de significacia (0.05)')
plt.legend()
plt.show()



