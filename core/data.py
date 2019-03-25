"""
Module for saving, loading and displaying data. To display the metadata of a
file, run this as a script.

Created on 2019-03-22
"""

import pickle
from tkinter import Tk, simpledialog
from tkinter.filedialog import askopenfilename, askopenfilenames, \
    asksaveasfilename
from datetime import datetime
from os.path import basename, join
from inspect import stack, getmodule
from os import getuid
from pwd import getpwuid
from constants import ROOT_DIR

USER_PATH = join(ROOT_DIR, 'data', 'users.pkl')


def save(file=None, data=None, metadata=None):
    """
    @author = Joel
    Save data and metadata to a file. Always set the first field of metadata
    as metadata={'description': info} (followed by other fields). Datetime of
    creation, the module that called save and the name of the user will be
    added to metadata automatically.

    Make sure to include samples, runtime, which Hamiltonian, minimizer,
    mattrix_to_op, ansatz, etc. was used when producing results. This should
    be done programmatically! E.g metadata = {'ansatz': ansatz_.__name__, ...}
    where ansatz_ is the ansatz being used.

    :param string file: file to save to
    :param data: data to save
    :param dictionary  metadata: metadata describing the data
    :return: None
    """
    # TODO: Failsafe save ASAP (handle errors)
    # UI get file if is None.
    if file is None:
        tk = Tk()
        tk.withdraw()
        file = asksaveasfilename(parent=tk, filetypes=[('Pickled', '.pkl')],
                                 initialdir=join(ROOT_DIR, 'data'))
    if file is None:
        raise FileNotFoundError("Can't save without a file.")

    # Add some fields automatically to metadata
    if metadata is None:
        metadata = {}  # to not get mutable default value
    metadata.update({key: default() for key, default in
                     _metadata_defaults().items() if key not in
                     metadata})

    # Make sure that data is dictionary
    if data is None:
        data = {}
    if not isinstance(data, dict):
        data = {'data': data}

    # Save
    with open(file, 'wb') as file_:
        pickle.dump({'data': data, 'metadata': metadata}, file_)


def load(file=None):
    """
    @author = Joel
    Load data and metadata from a file.

    :param string file: file to load from
    :return: (data, metadata)
    :rtype: (Any, dictionary )
    """
    # UI get file if is None.
    if file is None:
        tk = Tk()
        tk.withdraw()
        file = askopenfilename(parent=tk, filetypes=[('Pickled', '.pkl')],
                               initialdir=join(ROOT_DIR, 'data'))
    if file is None:
        raise FileNotFoundError("Can't load without a file.")

    # Load
    with open(file, 'rb') as file_:
        raw = pickle.load(file_)
    return raw['data'], raw['metadata']


def display(files=None):
    """
    @author = Joel
    Used to display file(s).

    :param files: files to display
    :return:
    """
    if files is None:
        tk = Tk()
        tk.withdraw()
        files = askopenfilenames(parent=tk, filetypes=[('Pickled', '.pkl')],
                                 initialdir=join(ROOT_DIR, 'data'))
    elif isinstance(files, str):
        files = [files]
    for file in files:
        data, metadata = load(file)
        print(
            '\n\033[1m' + 'Metadata from: ' + '\033[0m\n\033[92m' + file
            + '\033[0m\n')
        for key, value in metadata.items():
            print('\033[4m' + key.replace('_', ' ').capitalize() + ':\033[0m')
            print(value + '\n')
        print('\033[4m' + 'Variables in data' + ':\033[0m')
        for key in data:
            print(key + '\n')


def init_users(name):
    """
    @author = Joel
    Creates the user file.

    :param string name: your name
    :return:
    """
    users = {getpwuid(getuid())[0]: name}
    metadata = {'description': 'File containing known users as {username: '
                               'name}.',
                'created_by': name,
                }
    save(USER_PATH, users, metadata)


def add_user(name):
    """
    @author = Joel
    Adds a user to the user file (or changes the name if already existing).

    :param name: your name
    :return:
    """
    users = load(USER_PATH)
    _add_user(name, getpwuid(getuid())[0], users)


def _add_user(name, user, users):
    """
    @author = Joel
    Internal function (to keep DRY) for adding user/changing name in users.

    :param name: name of the added user
    :param user: user to be added
    :param users: current users as returned by load(USER_PATH)
    :return:
    """
    if user in users[0] and name != users[0][user]:
        print('\033[93mWarning: changing the name of existing user.\033[0m')
    users[0][user] = name
    save(USER_PATH, data=users[0], metadata=users[1])


def _get_name():
    """
    @author = Joel
    Finds who is trying to use save (for metadata purposes)

    :return: author
    :rtype: string
    """
    user = getpwuid(getuid())[0]
    try:
        users = load(USER_PATH)
    except FileNotFoundError as e:
        e.strerror = 'No user file found. Check USER_PATH and run init_users.'
        raise e
    if user in users[0]:
        name = users[0][user]
    else:
        tk = Tk()
        tk.withdraw()
        name = simpledialog.askstring("I don't recognize you.",
                                      "What is your name?", parent=tk)
        _add_user(name, user, users)
    return name


def _metadata_defaults():
    """
    @author = Joel
    Lazy initialization of metadata dictionary with default fields. Note the
    lambda.

    :return: metadata
    :rtype: dictionary
    """
    def get_caller():
        try:
            caller = basename(getmodule(stack()[3][0]).__file__)
        except:
            caller = 'unknown'
        return caller
    return {'created_by': _get_name,
            'created_from': get_caller,
            'created_datetime': lambda: datetime.now().strftime("%Y-%m-%d, "
                                                                "%H:%M:%S"),
            }


'''
@author = Joel
'''
if __name__ == '__main__':
    display()
