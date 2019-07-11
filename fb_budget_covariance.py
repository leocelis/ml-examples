import pandas as pd
from numpy import corrcoef

# report from Facebook Ads Reporting (Day breakdown)
columns = ['Ad Set Budget',
           'Amount Spent (USD)',
           'Impressions',
           'Reach',
           'Clicks (All)',
           'Result Type',
           'Results',
           'Cost per Result']

df = pd.read_csv("fb_adset_budget.csv", index_col=None, usecols=columns).dropna()
dfl = df.loc[df['Result Type'] == 'Lead']  # only Lead result type

# correlation coeficient
budget_cpa = round(corrcoef(dfl['Ad Set Budget'], dfl['Cost per Result'])[1, 0], 2)
budget_volume = round(corrcoef(dfl['Ad Set Budget'], dfl['Results'])[1, 0], 2)
spent_cpa = round(corrcoef(dfl['Amount Spent (USD)'], dfl['Cost per Result'])[1, 0], 2)
spent_volume = round(corrcoef(dfl['Amount Spent (USD)'], dfl['Results'])[1, 0], 2)

print()
print("==================================================")
print("Correlation coefficients")
print("==================================================\n")
print("'Ad Set Budget' vs. CPA: {}\n".format(budget_cpa))
print("'Ad Set Budget' vs. Volume: {}\n".format(budget_volume))
print("'Amount Spent' vs. CPA: {}\n".format(spent_cpa))
print("'Amount Spent' vs. Volume: {}\n".format(spent_volume))
print("==================================================\n")
