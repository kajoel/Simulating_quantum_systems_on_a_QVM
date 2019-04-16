"""
Recipe for long runs. This makes sure to save often and keeps the data in a
single file.

@author = Joel
"""
from core.data import save, load
from os.path import join


directory = 'my_dir'  # directory to save to
file = 'my_file'  # file to save to

# Complete path to the saved file (relative to the data directory):
path = join(directory, file)

try:
    # Try to load the file (will raise FileNotFoundError if not existing)
    data, metadata = load(path)
except FileNotFoundError:
    # Initialize data and metadata
    data = {}
    metadata = {'description': 'my description',
                'count': 0}  # IMPORTANT!


# Loop trough some parameters with a counter.
count = 0
for x in range(3):  # hint: look at itertools.product (might be useful)
    count += 1
    if count > metadata['count']:
        # In this case the relevant simulation has not already been saved to the
        # file and is therefore run now.

        #  ## RUN SIMULATION(S)* AND SAVE RELEVANT INFO IN DATA AND METADATA ##
        # * you can run a few simulations if saving takes time but don't do
        #   to many since all will be lost in case of crash.
        data[str(x)] = x**2  # dummy line

        # Update metadata['count'] and save file:
        metadata['count'] = count
        save(path, data, metadata)
