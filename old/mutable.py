class Mutable:
    def __init__(self, value):
        super().__init__()
        self.value = value

    def set(self, value):
        self.value = value

    def __getattribute__(self, item):
        if item is 'value':
            return super().__getattribute__(item)
        elif item is 'set':
            return lambda x: Mutable.set(self, x)
        #elif item is '__class__':
        #    return Mutable
        else:
            print(item)
            return getattr(self.value, item)

    def type(self):
        return self.__class__

    def __str__(self):
        print('Hej')
        return self.value.__str__()
