from datetime import datetime
import numpy as np
import csv
from itertools import zip_longest, chain


csv.register_dialect(
    'mydialect',
    delimiter=',',
    quotechar='"',
    doublequote=True,
    skipinitialspace=True,
    lineterminator='\r\n',
    quoting=csv.QUOTE_MINIMAL)


def save(title, data, metadata):
    metadata.update({'time':'{}'.format(datetime.now())})
    print(str(metadata))

    with open(title + '.csv',"w+") as f:
        writer = csv.writer(f, dialect='mydialect')
        writer.writerow(str(metadata))
        writer.writerow(data.keys())
        for values in zip_longest(*data.values()):
            f1 = float(values[0])
            if values[1] is None: f2 = None
            else: f2 = float(values[1])
            writer.writerow((f1,f2))


def load(title):
    with open('test3.csv','r') as f:
        data_iter = csv.reader(f, delimiter = ',', quotechar = '"')
        info = next(data_iter, None)
        header = next(data_iter, None)
        data1 = [float(data[0]) for data in data_iter]


arr1 = list(np.ones((15,1)))
arr2 = list(2.3*np.ones((5,1)))

data = {'data1' : arr1, 'data2' : arr2}
metadata = {}

save('test4', data, metadata)
