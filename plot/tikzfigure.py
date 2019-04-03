from core import data
from constants import ROOT_DIR
from os.path import join
import matplotlib2tikz as tikz
import matplotlib.pyplot as plt

def create(title, datatitle):
    data_, metadata = data.load(datatitle)

    data_keys = data_.keys()

    if len(data_keys) == 2:
        x = data_keys[0]
        y = data_keys[1]
        plt.plot(x,y)
        tikz.save(title + ".tex")

###############################################################################
#TEST
###############################################################################
datatitle = join(ROOT_DIR,'data', '2DimSweepSampNone.pkl')

create('test', datatitle)