import tensorflow as tf
import pandas as pd
import os

def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

files = getListOfFiles('.')
gameList=['Pong','DemonAttack','MsPacman','Frostbite','BankHeist','Asterix']
attntype = ['NoAttn','Attn','CrossAttn','RvuAttn']
seeds = ['69', '71', '142']
data = {}
for attn in attntype:
    for file in files:
        if file[-2:] == '.0':
            for game in gameList:
                if game+attn in file:
                    print('Printing '+game+' log files...')
                    for seed in seeds:
                        if seed in file[:60]:
                            Step = []
                            Value = []
                            for e in tf.compat.v1.train.summary_iterator(file):
                                for v in e.summary.value:
                                    if v.tag == 'eval/mean_reward':
                                        Step.append(e.step)
                                        Value.append(v.simple_value)
                            df = pd.DataFrame({'Step':Step, 'Value': Value})
                            if attn+game+seed in data:
                                data[attn+game+seed+'run2'] = df
                                df.to_csv(attn+game+seed+'run2.csv', index = False, header=True)
                            else:
                                data[attn+game+seed] = df
                                df.to_csv(attn+game+seed+'.csv', index = False, header=True)
