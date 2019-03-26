"""
@author = Joel
Test file for inspecting path and making sure that import work.
"""

import sys
print('Python path:')
for x in sys.path:
    print(x)

from core import lipkin_quasi_spin
from constants import ROOT_DIR

print('\nThis should be the lipkin_quasi_spin module:\n'
      + str(lipkin_quasi_spin))
print('\nProject root dir:\n' + ROOT_DIR)
