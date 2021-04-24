import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
pd.set_option('display.max_rows', None)

curfiles = os.listdir()
curfiles = [y for y in curfiles if y[-1]=='v']

def getData(game, str, currfiles, smoothing):
    f = [x for x in currfiles if str in x and game in x]
    max = []
    for index, path in enumerate(f):
        if index == 0:
            df = pd.read_csv(path)
            df = df.loc[df['Step']<=5100192]
            max.append(df.loc[df['Step']<=5000192]['Value'].max())
            df['Value'] = df['Value'].ewm(alpha=(1 - smoothing)).mean()
        else:
            df2 = pd.read_csv(path)
            df2 = df2.loc[df2['Step']<=5100192]
            max.append(df2.loc[df2['Step']<=5000192]['Value'].max())
            df2['Value'] = df2['Value'].ewm(alpha=(1 - smoothing)).mean()
            df = pd.concat([df,df2])
        df[str] = sum(max)/len(max)
    return df

def aggData(game, datakwarg, datanames, currfiles, smoothing):
    for index, x in enumerate(datakwarg):
        if index == 0:
            df = 0
            df = getData(game, x, currfiles, smoothing)
            df['Model'] = datanames[index]
        else:
            df2 = getData(game, x, currfiles, smoothing)
            df2['Model'] = datanames[index]
            df = pd.concat([df,df2])
    return df

gamelist=['Alien','Amidar','Assault','Asterix','Asteroids','Atlantis','BankHeist','BattleZone','BeamRider',
'Bowling','Boxing','Breakout','Centipede','ChopperCommand','CrazyClimber','DemonAttack','DoubleDunk','Enduro',
'FishingDerby','Freeway','Frostbite','Gopher','Gravitar','IceHockey','Jamesbond','Kangaroo','Krull','KungFuMaster',
'MontezumaRevenge','MsPacman','NameThisGame','Pitfall','Pong','PrivateEye','Qbert','Riverraid','RoadRunner',
'Robotank','Seaquest','SpaceInvaders','StarGunner','Tennis','TimePilot','Tutankham','UpNDown','Venture',
'VideoPinball','WizardOfWor','Zaxxon']
col = 5
row=int(len(gamelist)/col+1)
data = {}

fig, axes = plt.subplots(nrows=row, ncols=col, sharex=True)
fig.set_size_inches(11.69,16)
for i in range(row):
    for j in range(col):
        if i*col+j<len(gamelist):
            game = gamelist[i*col+j]
            data[game] = aggData(game,['NoAttn','RvuAttn','SelfAttn','CrossAttn'], ['No Attn','C-SAN','SAN','CAN'], curfiles, 0.5)
            axes[i][j] = sns.lineplot(data=data[game], x="Step", y="Value", hue = "Model", style = "Model",ax=axes[i][j])
            axes[i][j].set_title(game)
            axes[i][j].set(ylabel=None)
            axes[i][j].set(xlabel=None)
            if game == gamelist[-1]:
                handles, labels = axes[i][j].get_legend_handles_labels()
                fig.legend(handles, labels, loc='lower right', fontsize=20)
            axes[i][j].get_legend().remove()
        else:
            fig.delaxes(axes[i][j])
plt.xlim(0,5000000)
plt.tight_layout()
fig.suptitle('Model Comparison Across ALE', position=(.5,1.1), fontsize=20)
fig.show()
fig.savefig('ALEPlot',dpi=300)

print('NoAttn,self,rvuAttn,CrossAttn')
for game in gamelist:
    NoAttn = data[game]['NoAttn'].max()
    RvuAttn = data[game]['RvuAttn'].max()
    SelfAttn = data[game]['SelfAttn'].max()
    CrossAttn = data[game]['CrossAttn'].max()

    print(str(round(NoAttn,2))+','+str(round(SelfAttn,2))+','+str(round(RvuAttn,2))+','+str(round(CrossAttn,2)))
