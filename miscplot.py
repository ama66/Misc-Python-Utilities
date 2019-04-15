## Plotting and animation 
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation, rc
from IPython.display import HTML
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'

import pandas as pd
import datetime
import zipfile
# Import the os module, for the os.walk function
import os
import sys
import json
import shutil
plt.style.use('ggplot')  
pd.options.mode.chained_assignment = None 
import seaborn as sns
import pickle
# import matplotlib.dates as mdates

fig, axs = plt.subplots(nrows=1, ncols=3,figsize=(15,6))
        
ax = axs[0]
ax.hist(df["var1"].values, bins=20);
ax.set_title('var1')

ax = axs[1]
ax.hist(df["var2"].values, bins=20);
ax.set_title('var2')


ax = axs[2]
ax.hist(df["var3"].values, bins=20);
ax.set_title('var3')


fig.suptitle(f'variable title in n={i}')

imagepath="variablepath"+str(i)+".png"
plt.savefig(imagepath)
## or fig.savefig
############################
 # ax.set_xlabel("")
## Seaborn 

import collections
plt.style.use('ggplot')  
data = np.random.permutation(np.array(["dog"]*10 + ["cat"]*7 + ["rabbit"]*3))
counts = collections.Counter(data)
plt.bar(range(len(counts)), list(counts.values()), tick_label=list(counts.keys()))
import io
import requests
import seaborn as sns

sns.set(style="whitegrid")
#tips = sns.load_dataset("tips")
tips= pd.read_csv("~/Documents/tips.csv")
tips.head()
## for dataframe 
ax = sns.barplot(x="day", y="total_bill", data=tips)


###

import collections
plt.style.use('ggplot')  

data = {"strongly disagree": 5,
        "slightly disagree": 3,
        "neutral": 8,
        "slightly agree": 12,
        "strongly agree": 9}
plt.bar(range(len(data)), data.values(),color=['black', 'red', 'green', 'blue', 'cyan'])


plt.xticks(range(len(data)), data.keys(), rotation=45, ha="right");

#### horizontal 

import collections
plt.style.use('ggplot')  

data = {"strongly disagree": 5,
        "slightly disagree": 3,
        "neutral": 8,
        "slightly agree": 12,
        "strongly agree": 9}
plt.barh(range(len(data)), list(data.values()),color=['b', 'red', 'green', 'blue', 'cyan'])


plt.yticks(range(len(data)), data.keys(), rotation=45, ha="right");

########### Timeseries plots 

ax1=df.plot(figsize=(15,8), style='o')
plt.legend(["firstelement"] )
plt.title('title')
plt.xlabel('somelab')
plt.ylabel("ylabel", fontsize=14)
ax1.axvline('xpoint',linestyle='--', linewidth=3 , color='black')

#### drawing text at some position

ax1.axhline(ypoint,linestyle='--', linewidth=3 , color='black',xmin=0.185)
textstr = f'some tex= {i}'
props = dict(boxstyle='round', facecolor='green', alpha=0.5)
ax1.text(0.05, 0.8, textstr, transform=ax1.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

######## Resample

dg=df.resample('1D').mean()
ax1=df.plot(figsize=(15,8), style='o')
df.rolling(4).mean().plot(ax=ax1, linewidth=5, fontsize=14, style='-')
#### Box plot sns 
sns.boxplot(x=df['var1'])


### Animations 

import numpy as np
import scipy as sp
import scipy.fftpack
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib._color_data as mcd
# import plotly.plotly as py
# import plotly.tools as tls
import matplotlib.dates as mdates
from matplotlib import animation, rc
from IPython.display import HTML
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
plt.style.use('ggplot')


Writer = animation.writers['ffmpeg']
writer = Writer(fps=0.5, metadata=dict(artist='Me'), bitrate=1800)
plt.ion()

## DF in global var space 
# initialization function: plot the background of each frame
# First set up the figure, the axis, and the plot element we want to animate

fig, ax = plt.subplots(figsize=(10,8));
line, = ax.plot([], [], lw=2,linestyle='',marker='o');
ax.set_ylim(0,8 );
ax.set_ylabel('y axis var')
ax.set_xlabel('time /x axis')

def init():
    line.set_data([], [])
    return (line,)

# animation function. This is called sequentially
def animate(date_str):
    x = DF[date_str].index.to_pydatetime()
    ax.set_xlim(np.min(x), np.max(x))
    y = DF[date_str].values
    line.set_data(x, y)
    #set major ticks format
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    ax.set_title(f'title at {date_str}')
    return (line,)

anim = animation.FuncAnimation(fig, animate,date_list, init_func=init,
                                interval=1500, blit=True)
HTML(anim.to_html5_video())
anim.save('./filename.mp4', writer=writer)




## Another Animation Example 
# https://github.com/markjay4k/fourier-transform/blob/master/Animated%20Sinc%20and%20FT%20example.ipynb

import numpy as np

def rect(x, B):
    """
    create a rectangle function
    returns a numpy array that is 1 if |x| < w and 0 if |x| > w
    B is the rectangle width centered at 0
    x is the number of points in the array
    """
    
    B = int(B)
    x = int(x)
    
    high = np.ones(B)
    low1 = np.zeros(int(x/2 - B/2))    
    x1 = np.append(low1, high)
    rect = np.append(x1, low1)
    
    if x > len(rect):
        rect = np.append(rect, 0)
    elif x < len(rect):
        rect = rect[:-1]

    return rect


%matplotlib notebook
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# constants and x array
pi = np.pi
length = 2000
x = np.linspace(-1, 1, length)

# create figure and axes 
fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 6))

# creating our line objects for the plots
sinc, = ax1.plot(x, np.sin(x), '-b')
box, = ax2.plot(x, np.sin(x), '-r')

def animate(B):
    """
    this function gets called by FuncAnimation
    each time called, it will replot with a different width "B"
    
    B: rect width
    
    return:
        sinc: ydata
        box: ydata
    """
    
    # create our rect object
    f = rect(len(x), B)
    box.set_ydata(f)
    
    # create our sinc object
    F = (B / length) * np.sin(x * B / 2) / (x * B / 2)
    sinc.set_ydata(F)
    
    # adjust the sinc plot height in a loop
    ax1.set_ylim(np.min(F), np.max(F))
    
    # format the ax1 yticks
    plt.setp(ax1, xticks=[-0.25, 0.25], xticklabels=['-1/4', '1/4'],
             yticks=[0, np.max(F)], yticklabels=['0', 'B={:.2f}'.format((B / length))])
    
    # format the ax2 xticks to move with the box
    plt.setp(ax2, yticks=[0, 1], 
             xticks=[-1, -1 * B / length, 1 * B / length, 1], xticklabels=['-1', '-B/2', 'B/2', '1'])
    
def init():
    """
    initialize the figure
    """
    
    ax2.set_ylim(-0.2, 1.1)
    ax1.set_xlim(-0.25, 0.25)
    ax2.set_xlim(-1, 1)
    ax1.axhline(0, color='black', lw=1)
    ax2.axhline(0, color='black', lw=1)
    plt.rcParams.update({'font.size':14})
    
    return sinc, box,

# the FuncAnimation function iterates through our animate function using the steps array
step = 10
steps = np.append(np.arange(10, 1000, step), np.arange(1000, 10, -1 * step))
ani = FuncAnimation(fig, animate, steps, init_func=init, interval=50, blit=True)
plt.show()







  

