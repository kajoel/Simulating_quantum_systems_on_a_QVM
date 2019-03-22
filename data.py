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
from os.path import basename, join, dirname
from inspect import stack, getmodule
from os import getuid
from pwd import getpwuid

USER_PATH = join(dirname(__file__), 'Data/users.pkl')


def save(file=None, data=None, metadata=None):
    """
    @author = Joel
    Save data and metadata to a file. Always use
    metadata={'description': info} (and possibly more fields such as samples,
    runtime, etc.)

    :param string file: file to save to
    :param data: data to save
    :param dictionary  metadata: metadata describing the data
    :return: None
    """
    # UI get file if is None.
    if file is None:
        tk = Tk()
        tk.withdraw()
        file = asksaveasfilename(parent=tk, filetypes=[('Pickled', '.pkl')],
                                 initialdir='./Data')
    if file is None:
        raise FileNotFoundError("Can't save without a file.")

    # Add some fields automatically to metadata
    if metadata is None:
        metadata = {}  # to not get mutable default value
    metadata.update({key: default() for key, default in
                     _metadata_defaults().items() if key not in
                     metadata})

    # Save
    with open(file, 'wb') as file_:
        pickle.dump({'data': data, 'metadata': metadata}, file_)


def load(file=None, loadkwargs=None):
    """
    @author = Joel
    Load data and metadata from a file.

    :param string file: file to load from
    :param dictionary loadkwargs: extra kwargs to askopenfilename
    :return: (data, metadata)
    :rtype: (Any, dictionary )
    """
    # UI get file if is None.
    if file is None:
        tk = Tk()
        tk.withdraw()
        if loadkwargs is None:
            loadkwargs = {}
        file = askopenfilename(parent=tk, filetypes=[('Pickled', '.pkl')],
                               initialdir='./Data', **loadkwargs)
    if file is None:
        raise FileNotFoundError("Can't load without a file.")

    # Load
    with open(file, 'rb') as file_:
        raw = pickle.load(file_)
    return raw['data'], raw['metadata']


def _metadata_defaults():
    """
    @author = Joel
    Lazy initialization of metadata dictionary with default fields. Note the
    lambdas.

    :return: metadata
    :rtype: dictionary
    """

    # TODO:
    # 'created_from': lambda: basename(getmodule(stack()[1][0]).__file__),
    return {'created_datetime': lambda: datetime.now().strftime("%Y-%m-%d, "
                                                                "%H:%M:%S"),
            'created_by': lambda: _get_user(),
            }


def _get_user():
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
        users[0][user] = name
        save(USER_PATH, data=users[0], metadata=users[1])
    return name


def init_users(username):
    """
    @author = Joel
    Creates the user file.

    :param string username: your username
    :return:
    """
    users = {getpwuid(getuid())[0]: username}
    metadata = {'description': 'File containing known users as {username: '
                               'alias}.',
                'created_by': username,
                }
    save(USER_PATH, users, metadata)


'''
@author = Joel
'''
if __name__ == '__main__':
    root = Tk()
    root.withdraw()
    files = askopenfilenames(parent=root, filetypes=[('Pickled', '.pkl')],
                             initialdir='./Data')
    for file in files:
        data, metadata = load(file)
        print(
            '\n\033[1m' + 'Metadata from: ' + '\033[0m\n\033[92m' + file
            + '\033[0m\n')
        for key, value in metadata.items():
            print('\033[4m' + key.replace('_', ' ').capitalize() + ':\033[0m')
            print(value + '\n')
