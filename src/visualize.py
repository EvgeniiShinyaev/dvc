import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('data/processed_iris.csv')

import os
os.makedirs('plots', exist_ok=True)
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.savefig('plots/correlation_matrix.png')  
plt.show()

sns.pairplot(df, hue="target", palette="husl")
plt.savefig('plots/pairplot.png')
plt.show()
