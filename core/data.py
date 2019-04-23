"""
Module for saving, loading and displaying data. To display the metadata of a
file, run this as a script.

Created on 2019-03-22
"""

import pickle
from warnings import warn
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askopenfilenames, \
    asksaveasfilename
from datetime import datetime
from os.path import basename, join, dirname, isdir, splitext
from inspect import stack, getmodule
from os import getuid, mkdir
from pwd import getpwuid
from constants import ROOT_DIR
from traceback import format_exc


BASE_DIR = join(ROOT_DIR, 'data')
USER_PATH = join(BASE_DIR, 'users.pkl')
HIGHEST_PROTOCOL = 2


def save(file=None, data=None, metadata=None,
         base_dir=BASE_DIR,
         force_extension=True,
         failsafe=True,
         protocol=None,
         extract=False,
         disp=True):
    """
    Save data and metadata to a file. Always set the first field of metadata
    as metadata={'description': info} (followed by other fields). Datetime of
    creation, the module that called save and the name of the user will be
    added to metadata automatically.

    Make sure to include samples, runtime, which Hamiltonian, minimizer,
    mattrix_to_op, ansatz, etc. was used when producing results. This should
    be done programmatically! E.g metadata = {'ansatz': ansatz_.__name__, ...}
    where ansatz_ is the ansatz being used.

    @author = Joel

    :param string file: file to save to
    :param string base_dir: prepended to file
    :param bool force_extension: if True save will add the extension '.pkl' if
        the file variable is missing extension
    :param data: data to save
    :param dictionary  metadata: metadata describing the data
    :param bool failsafe: determines whether to use failsafe save or not
    :param protocol: Protocol to use when saving. None (which results in
        HIGHEST_PROTOCOL) is recommended.
    :param bool extract: determines whether to append or extend data to file (
        for protocol>1)
    :param bool disp: if True display metadata after saving
    :return: None
    """
    # UI get file if is None and create directory if not existing.
    if file is None:
        tk = Tk()
        tk.withdraw()
        file = asksaveasfilename(parent=tk, filetypes=[('Pickled', '.pkl'),
                                                       ('All files', '.*')],
                                 initialdir=base_dir)
    else:
        file = join(base_dir, file)
        # Create directory (if not existing)
        dir_ = dirname(file)
        if not isdir(dir_):
            mkdir(dir_)
        # Add extension .pkl if extension is missing and force_extension
        if not splitext(file)[1] and force_extension:
            file = file + '.pkl'

    if protocol is None:
        protocol = HIGHEST_PROTOCOL

    if not isinstance(protocol, int) or protocol < 1\
            or protocol > HIGHEST_PROTOCOL:
        warn(f"Unsupported protocol ({protocol}). Using protocol"
             f"{HIGHEST_PROTOCOL} instead.")
        protocol = HIGHEST_PROTOCOL

    # Add some fields automatically to metadata
    if metadata is None:
        metadata = {}  # to not get mutable default value
    metadata.update({key: default() for key, default in
                     _metadata_defaults(protocol).items() if key not in
                     metadata})

    # Save
    if failsafe:
        success = _failsafe_save(file, data, metadata, protocol, extract)
    else:
        quick_save(file, data, metadata, protocol, extract)
        success = True

    # Display
    if success and disp:
        try:
            _display_internal(file, metadata)
        except:
            print('\033[93m' + 'The following error occurred while trying to '
                               'display metadata from the saved file:'
                  + '\033[0m\n')
            print('\033[91m' + format_exc() + '\033[0m')

    # Return file, data and metadata
    return file, data, metadata


def quick_save(file, data, metadata, protocol, *args):
    """
    This function should primarily be used after using save one time when
    saving to the same file multiple times. This function will not add fields
    to metadata and is not failsafe (keep metadata and file returned by save).

    @author = Joel

    :param string file: file to save to
    :param data: data to save
    :param dictionary  metadata: metadata describing the data
    :param int protocol: protocol
    :return:
    """
    if protocol == 1:
        _save_1(file, data, metadata, *args)
    elif protocol == 2:
        _save_2(file, data, metadata, *args)
    else:
        raise ValueError(f'Unsupported protocol ({protocol}).')


def append(file, data, base_dir=BASE_DIR, force_extension=True):
    """
    Append data to file. Make sure that the original file were saved with
    protocol 2.

    :param file: file to append to
    :param data: data to append
    :param string base_dir: prepended to file
    :param bool force_extension: if True save will add the extension '.pkl' if
        the file variable is missing extension
    :return:
    """
    file = join(base_dir, file)
    # Add extension .pkl if extension is missing and force_extension
    if not splitext(file)[1] and force_extension:
        file = file + '.pkl'

    with open(file, 'ab') as file_:
        _append_internal(data, file_)


def extend(file, data, base_dir=BASE_DIR, force_extension=True):
    """
    Extend file with data (extend and append have the same relationship as
    the corresponding functions for lists). Make sure that the original file
    were saved with protocol 2.

    :param file: file to extend
    :param data: data to extend file with
    :param string base_dir: prepended to file
    :param bool force_extension: if True save will add the extension '.pkl' if
        the file variable is missing extension
    :return:
    """
    file = join(base_dir, file)
    # Add extension .pkl if extension is missing and force_extension
    if not splitext(file)[1] and force_extension:
        file = file + '.pkl'

    with open(file, 'ab') as file_:
        _extend_internal(data, file_)


def load(file=None, base_dir=join(ROOT_DIR, 'data'), force_extension=True):
    """
    Load data and metadata from a file.

    @author = Joel

    :param string file: file to load from
    :param string base_dir: prepended to file
    :param bool force_extension: if True load will add the extension '.pkl' if
        the file variable is missing extension.
    :return: (data, metadata)
    :rtype: (Any, dictionary )
    """
    # UI get file if is None.
    if file is None:
        tk = Tk()
        tk.withdraw()
        file = askopenfilename(parent=tk, filetypes=[('Pickled', '.pkl'),
                                                     ('All files', '.*')],
                               initialdir=join(ROOT_DIR, 'data'))
    else:
        file = join(base_dir, file)
        # Add extension .pkl if extension is missing and force_extension
        if not splitext(file)[1] and force_extension:
            file = file + '.pkl'

    if file is None or file is ():
        raise FileNotFoundError("Can't load without a file.")

    # Load raw
    raw = _load_raw(file)

    # Determine protocol
    protocol = _protocol_determine(raw)

    # Format output based on protocol
    if protocol == 1:
        return _format_1(raw)
    elif protocol == 2:
        return _format_2(raw)
    else:
        warn(f"Can't format data with protocol {protocol}. Returning raw "
             f"content.")
        return raw


def display(files=None):
    """
    Used to display file(s).

    @author = Joel

    :param files: files to display
    :return:
    """
    if files is None:
        tk = Tk()
        tk.withdraw()
        files = askopenfilenames(parent=tk, filetypes=[('Pickled', '.pkl'),
                                                       ('All files', '.*')],
                                 initialdir=join(ROOT_DIR, 'data'))
    elif isinstance(files, str):
        files = [files]
    for file in files:
        data, metadata = load(file)
        _display_internal(file, metadata)


def init_users(name):
    """
    Creates the user file.

    @author = Joel

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
    Adds a user to the user file (or changes the name if already existing).

    @author = Joel

    :param name: your name
    :return:
    """
    users = load(USER_PATH)
    _add_user(name, getpwuid(getuid())[0], users)


def _add_user(name, user, users):
    """
    Internal function (to keep DRY) for adding user/changing name in users.

    @author = Joel

    :param name: name of the added user
    :param user: user to be added
    :param users: current users as returned by load(USER_PATH)
    :return:
    """
    if user in users[0] and name != users[0][user]:
        print('\033[93mWarning: changing the name of existing user.\033[0m')
    users[0][user] = name
    save(USER_PATH, data=users[0], metadata=users[1])


def _display_internal(file, metadata):
    """
    Internal method for displaying file with metadata.

    @author = Joel

    :param str file: path to the file
    :param dict metadata: metadata describing data
    """
    print(
        '\n\033[1m' + 'Metadata from: ' + '\033[0m\n\033[92m' + file
        + '\033[0m\n')
    for key, value in metadata.items():
        print('\033[4m' + key.replace('_', ' ') + ':\033[0m')
        print(str(value) + '\n')


def _get_name():
    """
    Finds who is trying to use save (for metadata purposes).

    @author = Joel

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
        name = input("I don't recognize you. What's your name?")
        _add_user(name, user, users)
    return name


def _metadata_defaults(protocol):
    """
    Lazy initialization of metadata dictionary with default fields. Note the
    lambda.

    @author = Joel

    :param int protocol: protocol
    :return: metadata
    :rtype: dictionary
    """

    def get_caller():
        try:
            caller = basename(getmodule(stack()[3][0]).__file__)
        except:
            caller = 'unknown'
        return caller

    return {'protocol': lambda: protocol,
            'created_by': _get_name,
            'created_from': get_caller,
            'created_datetime': lambda: datetime.now().strftime("%Y-%m-%d, "
                                                                "%H:%M:%S"),
            }


def _failsafe_save(file, data, metadata, *args):
    """
    Method for (hopefully) failsafe saving.

    @author = Joel

    :param str file: file to save to
    :param data: data to save
    :param metadata: metadata describing the data
    :param int protocol: protocol
    :return:
    """
    modifiable = ['file', 'data', 'metadata']
    locals_ = {'file': file, 'data': data, 'metadata': metadata}
    exit_ = False
    success = False
    while not exit_:
        try:
            quick_save(file, data, metadata, *args)
            success = True
        except Exception as e:
            locals_['e'] = e
            print('\n\033[1m\033[93m' + 'Failed while saving. Handing the '
                                        'control to you.' + '\033[0m')
            print('\033[93m' + 'The following error occurred and can be '
                               'accessed as e:' + '\033[0m\n')
            print('\033[91m' + format_exc() + '\033[0m')
            print('\033[94m\033[1m\033[4m' + 'Instructions:' + '\033[0m')
            print('\033[94m\033[1m' + '%retry' + '\033[0m\033[94m' +
                  ' will cause the program to try to save again' + '\033[0m')
            print('\033[94m\033[1m' + '%exit' + '\033[0m\033[94m' +
                  ' will cause the program to quit without saving' + '\033[0m')
            print('\033[94m\033[1m' + '%skip' + '\033[0m\033[94m' +
                  ' (as the only thing on any line) will cause the program to '
                  'not \n\texecute anything of what you have entered' +
                  '\033[0m')
            print('\033[94m\033[1m' + '[blank line]' + '\033[0m\033[94m' +
                  ' will cause the program to execute the above lines\n' +
                  '\t(if not containing %skip)' + '\033[0m')
            print('\n\033[92m' + "You have access to the following "
                                 "variables \n(the ones in bold will be "
                                 "modified in data.save's scope):\n"
                  + ''.join('\t\033[92m' + '\033[1m' * (key in modifiable) + key
                            + '\033[0m' for key in locals_)
                  + '\033[0m')
            retry = False
            while not retry and not exit_:
                lines = _read_input()
                skip_str = '\n%skip\n'
                retry_str = '\n%retry\n'
                exit_str = '\n%exit\n'
                skip = skip_str in lines
                retry = (retry_str in lines) and not skip
                exit_ = (exit_str in lines) and not retry and not skip
                lines = lines.replace(skip_str, '\n'). \
                    replace(retry_str, '\n').replace(exit_str, '\n')
                if not skip:
                    try:
                        exec(lines, {}, locals_)
                    except:
                        print('\n\033[1m\033[93m' + 'Failed while executing:' +
                              '\033[0m')
                        print('\033[91m' + format_exc() + '\033[0m')
                        print('\033[93m' + 'You can try again.' + '\033[0m')
                        retry = False
                        exit_ = False
            file = locals_['file']
            data = locals_['data']
            metadata = locals_['metadata']
        else:
            exit_ = True
    return success


def _read_input():
    """
    Reads multiple lines from input (until blank line).

    @author = Joel

    :return: inputted lines
    """
    lines = ''
    prompt = '>>> '
    line = input(prompt)
    prompt = '... '
    while line:
        lines += '\n' + line
        line = input(prompt)
    return lines + '\n'


def _load_raw(file):
    """
    Load everything from pickle-file.

    @author = Joel

    :param string file: File to load from
    :return: Everything in file
    :rtype: list
    """
    raw = []
    with open(file, 'rb') as file_:
        while True:
            try:
                raw.append(pickle.load(file_))
            except EOFError:
                return raw


def _protocol_determine(raw):
    """
    Determine which protocol was used when saving the file.

    :param list raw: returned by _load_raw
    :return: protocol used when file was saved
    :rtype: int
    """
    if not len(raw):
        raise ValueError("Can't determine protocol of empty file.")

    if 'protocol' not in raw[0]:
        return 1

    return raw[0]['protocol']


def _format_1(raw):
    """
    Format data with protocol 1.

    :param raw: returned by _load_raw
    :return: formatted data
    """
    data, metadata = raw[0]['data'], raw[0]['metadata']

    # Check for more content
    if len(raw) > 1:
        base_key = 'extra'
        key = base_key
        count = 0
        while key in metadata:
            key = base_key + f'_{count}'
            count += 1
        metadata[key] = raw[1:]
        warn('File contains extra information which will be returned in '
             f'metadata[{key}].')

    return data, metadata


def _format_2(raw):
    """
    Format data with protocol 2.

    :param raw: returned by _load_raw
    :return: formatted data
    """
    return raw[1:], raw[0]


def _save_1(file, data, metadata, *args):
    """
    Save with protocol 1.

    :param file: file to save to
    :param data: data to save
    :param metadata: metadata to save
    :return:
    """
    # Make sure that data is a dictionary
    if data is None:
        data = {}
    if not isinstance(data, dict):
        data = {'data': data}

    with open(file, 'wb') as file_:
        pickle.dump({'data': data, 'metadata': metadata}, file_)


def _save_2(file, data, metadata, extend, *args):
    """
    Save with protocol 2.

    :param file: file to save to
    :param data: data to save
    :param metadata: metadata to save
    :return:
    """
    with open(file, 'wb') as file_:
        pickle.dump(metadata, file_)
        if extend:
            _extend_internal(data, file_)
        else:
            _append_internal(data, file_)


def _append_internal(data, file):
    """
    Append to open file.

    :param data: data to append
    :param file: open file
    :return:
    """
    pickle.dump(data, file)


def _extend_internal(data, file):
    """
    Extend open file with data.

    :param data: data to extend with,
    :param file: open file
    :return:
    """
    for data_ in data:
        pickle.dump(data_, file)


'''
@author = Joel
'''
if __name__ == '__main__':
    display()
