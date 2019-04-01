class MutableCompare:
    """
    Mutable object for comparison.

    @author = Joel
    """
    def __init__(self, value):
        super().__init__()
        self._value = value

    def set(self, value):
        self._value = value

    def __eq__(self, other):
        return self._value == other

    def __ne__(self, other):
        return self._value != other

    def __lt__(self, other):
        return self._value < other

    def __gt__(self, other):
        return self._value > other

    def __le__(self, other):
        return self._value <= other

    def __ge__(self, other):
        return self._value >= other
