class MutableCompare:
    """
    Mutable object for comparison.

    @author = Joel
    """
    def __init__(self, value):
        super().__init__()
        self.value = value

    def set(self, value):
        self.value = value

    def __eq__(self, other):
        return self.value == other

    def __ne__(self, other):
        return self.value != other

    def __lt__(self, other):
        return self.value < other

    def __gt__(self, other):
        return self.value > other

    def __le__(self, other):
        return self.value <= other

    def __ge__(self, other):
        return self.value >= other

    def __str__(self):
        return self.value.__str__()

    def __repr__(self):
        return self.value.__repr__()

