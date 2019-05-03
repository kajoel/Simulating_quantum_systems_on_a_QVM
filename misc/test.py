import plotly.plotly as py

import plotly.graph_objs as go
import numpy as np
import plotly
import pandas as pd
import csv

# Read data from a csv
import plotly.plotly as py
import plotly.graph_objs as go

import numpy as np
import csv



a = [[-1, 0, 1]]
dat = [[43, 56, 88], [13, 14, 15]]
newdat = [[a[0][i]] + dat[i] for i in range(len(a))]
anew = [[None]+a[0]]
print(anew)

with open('CSVtest.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(anew)
    writer.writerows(newdat)

writeFile.close()
'''

x, y, z = np.random.multivariate_normal(np.array([0,0,0]), np.eye(3), 200).transpose()
print(x)
trace1 = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=12,
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5
        ),
        opacity=0.8
    )
)

x2, y2, z2 = np.random.multivariate_normal(np.array([0,0,0]), np.eye(3), 200).transpose()
trace2 = go.Scatter3d(
    x=x2,
    y=y2,
    z=z2,
    mode='markers',
    marker=dict(
        color='rgb(127, 127, 127)',
        size=12,
        symbol='circle',
        line=dict(
            color='rgb(204, 204, 204)',
            width=1
        ),
        opacity=0.9
    )
)
print(trace1)
data = [trace1, trace2]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='simple-3d-scatter')
'''
#plotly.offline.plot(fig)