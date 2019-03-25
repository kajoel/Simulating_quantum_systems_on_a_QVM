from datetime import datetime
import numpy as np


def CSV(title, data1,data2):
    #names = ['Player Name', 'Foo', 'Bar']
    #scores = ['Score', 250, 500]
    np.savetxt(title + '.csv',  np.column_stack((data1, data2)),
               header ='{}'.format(datetime.now()), delimiter=',', fmt='%1.3f')
    #'{}'.format(datetime.now()),
    #,fmt="%d"

data1 = np.ones((4, 1))
data2 = 2.2* np.ones((4, 1))
CSV('test', data1,data2)

#arr = np.loadtxt('test.csv', unpack=True)
#print(arr)

# importing the csv module
import csv

csv.register_dialect(
    'mydialect',
    delimiter=',',
    quotechar='"',
    doublequote=True,
    skipinitialspace=True,
    lineterminator='\r\n',
    quoting=csv.QUOTE_MINIMAL)

# my data rows as dictionary objects

mydict = [{'branch': 'COE', 'cgpa': '9.0', 'name': 'Nikhil', 'year': ''},
          {'branch': 'COE', 'cgpa': '9.1', 'name': 'Sanchit', 'year': '2'},
          {'branch': 'IT', 'cgpa': '9.3', 'name': 'Aditya', 'year': '2'},
          {'branch': 'SE', 'cgpa': '9.5', 'name': 'Sagar', 'year': '1'},
          {'branch': 'MCE', 'cgpa': '7.8', 'name': 'Prateek', 'year': '3'},
          {'branch': 'EP', 'cgpa': '9.1', 'name': 'Sahil', 'year': '2'}]


#mydict = {'data1': np.array([1,0,1]), 'data2': np.array([0,1,0])}

# field names
fields = ['name', 'branch', 'year', 'cgpa']

# name of csv file
filename = "test2.csv"

# writing to csv file
with open(filename, 'w') as csvfile:
    # creating a csv dict writer object
    writer = csv.DictWriter(csvfile, fieldnames=fields, dialect='mydialect')

    # writing headers (field names)
    writer.writeheader()

    # writing data rows
    writer.writerows(mydict)


from itertools import zip_longest, chain
arr1 = list(np.ones((15,1)))
arr2 = list(2.2*np.ones((5,1)))

d = [arr1, arr2]
fields = ['data1', 'data2']

with open("test3.csv","w+") as f:
    writer = csv.writer(f, dialect='mydialect')
    writer.writerow(['info om denna filen'])
    writer.writerow((fields[0], fields[1]))
    for values in zip_longest(*d):
        f1 = float(values[0])
        if values[1] is None: f2 = None
        else: f2 = float(values[1])
        writer.writerow((f1,f2))

with open('test3.csv','r') as f:
    data_iter = csv.reader(f, delimiter = ',', quotechar = '"')
    info = next(data_iter, None)
    header = next(data_iter, None)
    data1 = [float(data[0]) for data in data_iter]

data1 = np.asarray(data1, dtype=float)
print(header)
