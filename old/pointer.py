from typing import Iterable


class Pointer:
    def __init__(self, value):
        self.__value = value

    def __getattribute__(self, key):
        if key == '__class__':
            return Pointer
        elif key == '_' + str(Pointer.__name__) + '__value':
            return super().__getattribute__(key)
        else:
            return getattr(self.__value, key)

    def __setattr__(self, key, value):
        if key == '_' + str(Pointer.__name__) + '__value':
            return super().__setattr__(key, value)
        else:
            return setattr(self.__value, key, value)

    def __delattr__(self, key: str) -> None:
        delattr(self.__value, key)

    def __reduce__(self) -> tuple:
        return self.__class__, (self.__value,)

    def __dir__(self) -> Iterable[str]:
        return dir(self.__value) + ['_' + str(Pointer.__name__) + '__value']

    def __eq__(self, other):
        return self.__value == other

    def __ne__(self, other):
        return self.__value != other

    def __str__(self):
        return self.__value.__str__()

    def __repr__(self):
        return self.__value.__repr__()

    def __format__(self, format_spec):
        return self.__value.__format__(format_spec)


    # TODO
    # def __eq__(self, o: object) -> bool:
    #     return super().__eq__(o)
    #
    # def __ne__(self, o: object) -> bool:
    #     return super().__ne__(o)
    #
    # def __str__(self) -> str:
    #     return super().__str__()
    #
    # def __repr__(self) -> str:
    #     return super().__repr__()
    #
    # def __format__(self, format_spec: str) -> str:
    #     return super().__format__(format_spec)
    #
    #
    #
    #
    #
    #



if __name__ == '__main__':
    a = Pointer(3)
