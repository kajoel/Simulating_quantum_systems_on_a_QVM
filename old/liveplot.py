import matplotlib.pyplot as plt
import numpy as np


class Liveplot:

    def __init__(self):
        self.figure = plt.figure()
        self.line = plt.plot([], [])[0]

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
