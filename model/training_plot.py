import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.ticker as mtick

from termcolor import colored

report = pd.read_excel('data/result_history/_training_report.xlsx')
#just to remove unnamed column
report = report.loc[:, ~report.columns.str.contains('^Unnamed')]
print(report)

#reshape dataframe
df_sns = pd.melt(report, id_vars="epoch", var_name="loss_acc", value_name="percentage")
plt.style.use('ggplot')
sns_fig = sns.lineplot(x='epoch', y='percentage', hue='loss_acc', data=df_sns, palette="muted")


sns_fig.figure.savefig('static/figures/training_report.png', dpi=600)
print(colored("Figure saved", "green"))