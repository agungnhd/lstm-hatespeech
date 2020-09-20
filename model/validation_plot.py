import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.ticker as mtick

from termcolor import colored

report = pd.read_excel('data/result_history/_validation_report.xlsx')
#just to remove unnamed column
report = report.loc[:, ~report.columns.str.contains('^Unnamed')]
print(report)

#reshape dataframe
df_sns = pd.melt(report, id_vars="K", var_name="metrics", value_name="percentage")
plt.style.use('ggplot')
sns_fig = sns.catplot(x='K', y='percentage', hue='metrics', data=df_sns, kind='bar', palette="muted")


sns_fig.savefig('static/figures/validation_report.png', dpi=600)
print(colored("Figure saved", "green"))