from core.data import load, save
from os.path import splitext


def split(file=None):
    data, metadata, file = load(file=file, return_path=True)
    name, ext = splitext(file)
    split_num = int(len(data)/2)
    save(file=name + '_part_1', data=data[:split_num], metadata=metadata,
         extract=True, base_dir='')
    save(file=name + '_part_2', data=data[:split_num], metadata=metadata,
         extract=True, base_dir='')


if __name__ == '__main__':
    split()


