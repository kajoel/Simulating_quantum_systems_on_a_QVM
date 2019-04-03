
"""
Dynamic function tolerance for Nelder-Mead Algorithm
"""


class DynFatol:

    def __init__(self, value):
        self.value = value

    def __ge__(self, other):
        print(other)
        print('ge')
        return self.value >= other

    def __le__(self, other):
        print('le')
        return self.value <= other

    def vals(self):
        print(self.value)
