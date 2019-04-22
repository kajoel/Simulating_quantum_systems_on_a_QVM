"""
Test of pickle-dumping.
"""
from constants import ROOT_DIR
from os.path import join
import pickle

file = join(ROOT_DIR, 'data', 'test_recipe', 'test_dump.pkl')

with open(file, 'ab') as f:
    pickle.dump([1, 2, 3], f)

