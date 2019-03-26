import matplotlib.pyplot as plt
import numpy as np
import math
import misc.compare.lib_joel as lib


# NOTE: The dataplotter tries to interpret the values. This can lead to weird behaviour in
#          edge-cases. If uncertain:
#           - initialize without x and y
#           - allways use x and y:s with deepth 3
#                  (e.g list of lists of lists or numpy arrays with ndim=3)


class dataplotter:
    countdataplotters = 0

    def __init__(self, x=None, y=None, labels=None, nbrlinesperplot=None,
                 tpause=0.01, tightlayout=True, plotlayout=None, name=None,
                 # Misc
                 colormap='hsv', nbrcolors=None, maxnbrcolors=8, markers=None,
                 # Colors and markers
                 legendplot=None, legendsize=(1, 1), showlegend=None,  # Legend
                 plottype='linear', plotlabels=None, showplotlabels=None,
                 # (Sub)plots
                 nbrfigures=None, figurelabels=None, showfigurelabels=None,
                 # Figures
                 figures=None, **kwargs):

        dataplotter.countdataplotters += 1

        ############################################################################################
        # INPUT CHECKS AND KWARG HANDLING
        ############################################################################################
        if x is not None:
            x = dataplotter._format_type(x)
        if y is not None:
            y = dataplotter._format_type(y)

        # INPUT CHECKS
        self.tpause = tpause

        # Defining:
        # self.nbrfigures, -.nbrplotsperfigure, -.nbrlinesperplot
        if nbrlinesperplot is not None:
            self.nbrlinesperplot = nbrlinesperplot
        elif labels is not None:
            pass  # see 15 lines further down
        elif y is not None:
            temp = lib.subscriptdepth(y)
            if temp == 0:
                self.nbrlinesperplot = 0
            elif temp == 1:
                self.nbrlinesperplot = len(y)
            else:
                self.nbrlinesperplot = len(y[0])
        else:
            raise ValueError(
                "Number of lines per plot was not specified explicitly nor " +
                "implicitly. Specify either nbrlinesperplot, labels or " +
                "initial y values")

        if type(labels) is not str and not lib.subscriptdepth(labels):
            if labels is None:
                self.labels = [''] * self.nbrlinesperplot
                if showlegend is None:
                    showlegend = False
            else:
                # DEBUG
                # print(type(labels))
                # print(lib.subscriptdepth(labels))
                # print((type(labels) is not str))
                # print((not lib.subscriptdepth(labels)))
                ###
                raise TypeError("labels should be a string or subscriptable")
        else:
            if type(labels) is str:
                self.labels = [labels]
            else:
                self.labels = labels
        if labels is not None and nbrlinesperplot is None:
            self.nbrlinesperplot = len(self.labels)

        if showlegend is None:
            showlegend = True

        if not lib.subscriptdepth(plottype) or type(plottype) is str:
            plottype = [plottype]
        self.plottype = []
        for temp in plottype:
            if callable(temp):
                self.plottype.append(temp)
            elif temp == 'linear':
                self.plottype.append(lambda x, y: (x, y))
            elif temp == 'loglog':
                self.plottype.append(
                    lambda x, y: (np.log10(x).tolist(), np.log10(y).tolist()))
            elif temp == 'logx':
                self.plottype.append(lambda x, y: (np.log10(x).tolist(), y))
            elif temp == 'logy':
                self.plottype.append(lambda x, y: (x, np.log10(y).tolist()))
            else:
                raise ValueError('plottype should be or contain functions ' +
                                 'and/or "linear", "loglog", "logx", "logy"')
        self.nbrplotsperfigure = len(self.plottype)

        if nbrfigures is None:
            if figurelabels is not None:
                self.nbrfigures = len(figurelabels)
            elif y is not None:
                temp = lib.subscriptdepth(y)
                if temp == 0:
                    self.nbrfigures = 1
                elif temp == 1:
                    if len(y) == self.nbrlinesperplot:
                        self.nbrfigures = 1
                    elif lib.subscriptdepth(x) and len(y) == len(x):
                        self.nbrfigures = 1
                    else:
                        self.nbrfigures = len(y)
                else:
                    self.nbrfigures = len(y)
            else:
                raise ValueError("The number of figures were not specified " +
                                 "explicitly nor implicitly")
        else:
            try:
                self.nbrfigures = int(nbrfigures)
                if self.nbrfigures != nbrfigures:
                    print(
                        "Warning! nbrfigures was not an integer and have been typecasted")
            except:
                raise ValueError(
                    "nbrfigures could not be interpreted as an integer")
        # self.nbrfigures, -.nbrplotsperfigure, -.nbrlinesperplot
        # should all be defined by now

        # More input checks and defaults

        if plotlabels is None:
            plotlabels = []
            for i in plottype:
                if type(i) is str:
                    plotlabels.append(i)
                elif callable(i):
                    plotlabels.append(i.__name__)
        elif not lib.subscriptdepth(plotlabels):
            plotlabels = [plotlabels]
        if len(plotlabels) == self.nbrplotsperfigure:
            self.plotlabels = plotlabels
        else:
            raise ValueError("len(plotlabels) != len(plottypes)")

        if showplotlabels is None:
            if self.nbrplotsperfigure == 1:
                showplotlabels = False
            else:
                showplotlabels = True

        try:
            if plotlayout[0] * plotlayout[1] < self.nbrplotsperfigure:
                raise ValueError("plotlayout is too small")
            elif any(type(plotlayout[i]) is not int for i in [0, 1]):
                raise TypeError("plotlayout contains non ints")
            else:
                self.plotlayout = plotlayout
        except (TypeError, IndexError):
            if plotlayout is None:
                temp = math.ceil(np.sqrt(self.nbrplotsperfigure))
                if temp * (temp - 1) < self.nbrplotsperfigure or temp == 0:
                    self.plotlayout = [temp, temp]
                else:
                    self.plotlayout = [temp - 1, temp]
            else:
                raise ValueError("plotlayout is not iterable, default (None)" +
                                 " or contains too few elements")

        if name is None:
            self.name = "Dataplotter " + str(dataplotter.countdataplotters)
        else:
            self.name = str(name)

        if figurelabels is not None and showfigurelabels is None:
            showfigurelabels = True
        elif showfigurelabels is None:
            showfigurelabels = False
        if type(figurelabels) is str:
            figurelabels = [figurelabels]
        if not lib.subscriptdepth(figurelabels):
            figurelabels = []
        initlength = len(figurelabels)
        for i in range(self.nbrfigures - initlength):
            figurelabels.append("Plot " + str(1 + i + initlength))
        if showfigurelabels:
            rect = [0, 0, 1, 0.95]
        else:
            rect = [0, 0, 1, 1]

        # KWARG HANDLING
        tempdict = {'linestyle': ':'}
        for key, value in tempdict.items():
            if key not in kwargs:
                kwargs[key] = value

        if nbrcolors is None:
            nbrcolors = len(self.labels)
        try:
            nbrcolors = min(nbrcolors, maxnbrcolors, float('inf'))
        except TypeError:
            raise ValueError(
                "nbrcolors and maxnbrcolors should be integers or default")

        try:
            self._colors = plt.cm.get_cmap(colormap, nbrcolors + 1)
            self.colormap = lambda x: self._colors(x % nbrcolors)
        except:
            if type(colormap) is np.ndarray:
                colormap = np.ndarray.tolist(colormap)
            if lib.subscriptdepth(colormap):
                self._colors = colormap
                self.colormap = lambda x: self._colors[x % nbrcolors]
            elif callable(colormap):
                self.colormap = lambda x: colormap(x % nbrcolors)
            else:
                raise ValueError(
                    'Colormap could not be interpreted by pyplot.cm.get_cmap, ' +
                    'is not an ndarray or iterable nor a callable')

        if markers is None:
            markers = ['o', 's', 'v', '^', '<', '>', 'p', 'd']
        elif type(markers) is not list:
            markers = [markers]
        self.markers = lambda x: markers[(x // nbrcolors) % len(markers)]

        if legendplot is None and self.nbrfigures == 1:
            legendplot = [0, 0]

        for temp in ['c', 'marker']:
            if temp in kwargs:
                del kwargs[temp]
        self.kwargs = kwargs

        ############################################################################################
        # FIGURE INIT
        ############################################################################################

        self.figures = figures

        self.lines = []
        templeg = []
        if figures is None:
            self.figures = []
            for i in range(self.nbrfigures):
                num = self.name + ": " + figurelabels[i]
                temp = plt.subplots(self.plotlayout[0], self.plotlayout[1],
                                    num=num)
                if type(temp[1]) is not np.ndarray:
                    temp = (temp[0],) + ((temp[1],),)
                else:
                    temp = (temp[0], temp[1].ravel())
                for j in range(self.plotlayout[0] * self.plotlayout[1],
                               self.nbrplotsperfigure, -1):
                    temp[1][j - 1].remove()
                self.figures.append(temp)
                if showfigurelabels:
                    self.figures[i][0].suptitle(figurelabels[i])
                line_j = []
                for j in range(self.nbrplotsperfigure):
                    if showplotlabels:
                        self.figures[i][1][j].set_title(self.plotlabels[j])
                    line_k = []
                    for k in range(self.nbrlinesperplot):
                        line, = self.figures[i][1][j].plot([None], [None],
                                                           c=self.colormap(k),
                                                           marker=self.markers(
                                                               k),
                                                           **self.kwargs)
                        if i == 0 and j == 0:
                            templeg.append(line)
                        line_k.append(line)
                    line_j.append(line_k)
                self.lines.append(line_j)
                if tightlayout:
                    self.figures[i][0].tight_layout(rect=rect)
                self.figures[-1][0].show()
        else: self.figures[-1][0].show()

        if showlegend:
            try:
                self.figures[legendplot[0]][1][
                    legendplot[1]].legend(templeg, self.labels,
                                          loc='upper left')
                self.figures[legendplot[0]][0].show()
            except:
                self.figlegend = plt.figure(
                    "Dataplotter " + str(dataplotter.countdataplotters)
                    + ": Legend", figsize=legendsize)
                self.figlegend.legend(templeg, self.labels, loc='center')
                self.figlegend.show()
        plt.pause(self.tpause)
        if x is not None and y is not None:
            self.addValues(x, y)

    ################################################################################################
    # ADDVALUES AND OTHER CLASS FUNCTIONS
    ################################################################################################
    def addValues(self, x, y):
        # DEBUG
        # print('First in addValues')
        # print('x:')
        # print(x)
        # print('y:')
        # print(y)
        ###

        (x, y) = self.format(x, y)

        # DEBUG
        # print('Second in addValues')
        # print('x:')
        # print(x)
        # print('y:')
        # print(y)
        ###

        # Plot
        for i in range(self.nbrfigures):
            for j in range(self.nbrplotsperfigure):
                for k in range(self.nbrlinesperplot):
                    (tempx, tempy) = self.plottype[j](x[i][k], y[i][k])
                    (oldx, oldy) = self.lines[i][j][k].get_data()
                    if not lib.iterable(tempx):
                        tempx = [tempx]
                    if not lib.iterable(tempy):
                        tempy = [tempy]
                    tempx = np.concatenate((oldx, tempx))
                    tempy = np.concatenate((oldy, tempy))
                    self.lines[i][j][k].set_data(tempx, tempy)
                self.figures[i][1][j].relim()
                self.figures[i][1][j].autoscale_view()
                # Might want: autoscale_view(True)
            self.figures[i][0].show()
        plt.pause(self.tpause)

    def format(self, x, y):
        # DEBUG
        # print("first format")
        # print(y)
        ###
        y = dataplotter._format_type(y)
        x = dataplotter._format_type(x)
        # DEBUG
        # print("second format")
        # print(y)
        ###

        y = self._format_shape(y)
        # DEBUG
        # print("third format")
        # print(y)
        ###
        try:
            nbrvalues = lib.recursive_len(y[0][0])
        except IndexError as e:
            if str(e) == "list index out of range":
                nbrvalues = 0

        if lib.recursive_len(x) != lib.recursive_len(y):
            depth = lib.subscriptdepth(x)
            if depth == 0:
                x = [[[x] * nbrvalues] * self.nbrlinesperplot] * self.nbrfigures
            elif depth == 1:
                if len(x) == nbrvalues:
                    x = [[x] * self.nbrlinesperplot] * self.nbrfigures
                elif len(x) == self.nbrlinesperplot:
                    x = [[[z] * nbrvalues for z in x]] * self.nbrfigures
                elif len(x) == self.nbrfigures:
                    x = [[[z] * nbrvalues] * self.nbrlinesperplot for z in x]
                else:
                    raise ValueError(
                        "x has 'depth' 1 but len(x) don't seem to match the " +
                        "dataplotter")
            elif depth == 2:
                if len(x) == self.nbrfigures:
                    if len(x[0]) == nbrvalues:
                        x = [[[z] * self.nbrplotsperfigure] for z in x]
                    elif len(x[0]) == self.nbrlinesperplot:
                        x = [[[w] * nbrvalues for w in z] for z in x]
                    else:
                        raise ValueError(
                            "x has 'depth' 2 and len(x) == numberoffigures " +
                            "but len(x[0] don't seem to match the dataplotter")
                elif len(x) == self.nbrlinesperplot and len(x[0]) == nbrvalues:
                    x = [x] * self.nbrplots
                else:
                    raise ValueError(
                        "x has 'depth' 2 but len(x) don't seem to match the " +
                        "dataplotter")
            else:
                raise ValueError(
                    "x and y don't seem to be compatible... but I'm not sure")
        else:
            x = self._format_shape(x)
        return (x, y)

    def _format_type(x):
        if type(x) is np.ndarray:
            if x.ndim == 1:
                return list(x)
            else:
                return [list(z) for z in x]
        else:
            return x

    def _format_shape(self, x):
        # DEBUG
        # print("_format_shape")
        # print(x)
        # print(self.nbrfigures)
        # print(self.nbrlinesperplot)
        ###
        depth = lib.subscriptdepth(x)
        if depth == 0:
            return [[x]]
        elif depth == 1:
            if self.nbrfigures == 1:
                return [x]
            elif self.nbrlinesperplot == 1:
                return [[z] for z in x]
        elif depth == 2:
            if len(x) == self.nbrlinesperplot:
                return [x]
            elif len(x) == self.nbrfigures and len(
                    x[0]) == self.nbrlinesperplot:
                return x
            elif len(x) == self.nbrfigures and len(
                    x[0]) != self.nbrlinesperplot:
                return [[z] for z in x]
        else:
            return x
