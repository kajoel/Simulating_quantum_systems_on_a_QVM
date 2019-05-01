import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm


class Liveplot:

    def __init__(self):
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.line = plt.plot([], [])[0]
        self.xvec = []
        self.yvec = []
        self.zvec = []

    def plotline(self, x, y, _=None):
        (oldx, oldy) = self.line.get_data()
        allx = np.concatenate((oldx, [x]))
        ally = np.concatenate((oldy, [y]))
        self.line.set_data(allx, ally)
        self.figure.axes[0].relim()
        self.figure.axes[0].autoscale_view()
        plt.pause(0.05)

    def scatter(self, x, y, _=None):
        plt.scatter(x, y, color='blue')
        plt.pause(0.05)

    '''
    def scatter3d(self, x, y, z, _=None):
        self.ax.scatter(x, y, z, color='blue')
        plt.pause(0.05)
    '''

    def scatter3d(self, **kwargs):
        def callback(x, y, *_, **__):
            #print('params:', x)
            #print('val:', y)
            xtmp = [x[i][0] for i in range(len(self.xvec), len(x))]
            ytmp = [x[i][1] for i in range(len(self.yvec), len(x))]
            ztmp = [y[i] for i in range(len(self.zvec), len(y))]
            #print('xlen:', len(self.xvec))
            #print('xtmp:', xtmp)
            #print('ytmp:', ytmp)
            #print('ztmp:', ztmp)
            self.xvec.extend(xtmp)
            self.yvec.extend(ytmp)
            self.zvec.extend(ztmp)
            #print('xvec:', self.xvec)
            # plt.clf()
            self.ax.scatter(self.xvec, self.yvec, self.zvec, color='blue',
                            **kwargs)
            plt.pause(0.05)

        return callback
