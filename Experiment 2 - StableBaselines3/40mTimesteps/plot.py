import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
pd.set_option('display.max_rows', None)

curfiles = os.listdir()
curfiles = [y for y in curfiles if y[-1]=='v']
def getData(str, currfiles, smoothing):
    f = [x for x in currfiles if str in x]

    for index, path in enumerate(f):
        if index == 0:
            df = pd.read_csv(path)
            df['Value'] = df['Value'].ewm(alpha=(1 - smoothing)).mean()
        else:
            df2 = pd.read_csv(path)
            df2['Value'] = df2['Value'].ewm(alpha=(1 - smoothing)).mean()
            df = pd.concat([df,df2])
    return df

def aggData(datakwarg, datanames, currfiles, smoothing):
    for index, x in enumerate(datakwarg):
        if index == 0:
            df = 0
            df = getData(x, currfiles, smoothing)
            df['Model'] = datanames[index]
        else:
            df2 = getData(x, currfiles, smoothing)
            df2['Model'] = datanames[index]
            df = pd.concat([df,df2])
    return df
game = 'DemonAttack'
df = aggData(['NoAttn'+game,'Attn'+game,'RvuAttn'+game,'CrossAttn'+game], ['No Attn','SAN','C-SAN','CAN'], curfiles, 0.5)

plot1 = sns.lineplot(data=df, x="Step", y="Value", hue = "Model", style = "Model")
plt.title(game, fontsize = 15)
plt.ylabel("Score", fontsize = 12)
plt.xlabel("No. Epochs", fontsize = 12)
plt.xlim(0,40000000)
plt.figure(figsize = (16,9))
plt.legend(loc='upper left')
plot1.figure.savefig(game,dpi=300)

fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True)
game = 'Asterix'
data1 = aggData(['NoAttn'+game,'Attn'+game,'RvuAttn'+game,'CrossAttn'+game], ['No Attn','SAN','C-SAN','CAN'], curfiles, 0.5)
axes[0][0] = sns.lineplot(data=data1, x="Step", y="Value", hue = "Model", style = "Model",ax=axes[0][0])
axes[0][0].set_title(game)
axes[0][0].set(ylabel=None)
axes[0][0].set(xlabel=None)
axes[0][0].get_legend().remove()
game = 'BankHeist'
data2 = aggData(['NoAttn'+game,'Attn'+game,'RvuAttn'+game,'CrossAttn'+game], ['No Attn','SAN','C-SAN','CAN'], curfiles, 0.5)
axes[0][1] = sns.lineplot(data=data2, x="Step", y="Value", hue = "Model", style = "Model",ax=axes[0][1])
axes[0][1].set_title(game)
axes[0][1].set(ylabel=None)
axes[0][1].set(xlabel=None)
axes[0][1].get_legend().remove()
game = 'Frostbite'
data3 = aggData(['NoAttn'+game,'Attn'+game,'RvuAttn'+game,'CrossAttn'+game], ['No Attn','SAN','C-SAN','CAN'], curfiles, 0.5)
axes[1][0] = sns.lineplot(data=data3, x="Step", y="Value", hue = "Model", style = "Model",ax=axes[1][0])
axes[1][0].set_title(game)
axes[1][0].set(ylabel=None)
axes[1][0].set(xlabel=None)
axes[1][0].get_legend().remove()
game = 'MsPacman'
data4 = aggData(['NoAttn'+game,'Attn'+game,'RvuAttn'+game,'CrossAttn'+game], ['No Attn','SAN','C-SAN','CAN'], curfiles, 0.5)
axes[1][1] = sns.lineplot(data=data4, x="Step", y="Value", hue = "Model", style = "Model",ax=axes[1][1])
axes[1][1].set_title(game)
axes[1][1].set(ylabel=None)
axes[1][1].set(xlabel=None)
axes[1][1].get_legend().remove()
plt.xlim(0,40000000)
plt.tight_layout()
fig.show()
fig.savefig('Comparison',dpi=300)
