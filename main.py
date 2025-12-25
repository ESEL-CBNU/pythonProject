import seaborn as sns
import matplotlib.pyplot as plt
from warnings import filterwarnings
df = sns.load_dataset('tips')
sns.histplot(df['total_bill'], kde = True, color ='green', bins = 20)
sns.jointplot(x ='total_bill',color ='green', y ='tip', data = df)