from typing import Iterable, Any, Optional, Tuple, Sequence


class MutableNum(int):
    def __init__(self, value) -> None:
        super().__init__()
        self.value = value

    def bit_length(self) -> int:
        return self.value.bit_length()

    def to_bytes(self, length: int, byteorder: str, *,
                 signed: bool = ...) -> bytes:
        return self.value.to_bytes(length, byteorder, signed=signed)

    @classmethod
    def from_bytes(cls, bytes: Sequence[int], byteorder: str, *,
                   signed: bool = ...) -> int:
        return int.from_bytes(bytes, byteorder, signed=signed)

    def __add__(self, x: int) -> int:
        return self.value.__add__(x)

    def __sub__(self, x: int) -> int:
        return self.value.__sub__(x)

    def __mul__(self, x: int) -> int:
        return self.value.__mul__(x)

    def __floordiv__(self, x: int) -> int:
        return self.value.__floordiv__(x)

    def __truediv__(self, x: int) -> float:
        return self.value.__truediv__(x)

    def __mod__(self, x: int) -> int:
        return self.value.__mod__(x)

    def __divmod__(self, x: int) -> Tuple[int, int]:
        return self.value.__divmod__(x)

    def __radd__(self, x: int) -> int:
        return self.value.__radd__(x)

    def __rsub__(self, x: int) -> int:
        return self.value.__rsub__(x)

    def __rmul__(self, x: int) -> int:
        return self.value.__rmul__(x)

    def __rfloordiv__(self, x: int) -> int:
        return self.value.__rfloordiv__(x)

    def __rtruediv__(self, x: int) -> float:
        return self.value.__rtruediv__(x)

    def __rmod__(self, x: int) -> int:
        return self.value.__rmod__(x)

    def __rdivmod__(self, x: int) -> Tuple[int, int]:
        return self.value.__rdivmod__(x)

    def __pow__(self, x: int) -> Any:
        return self.value.__pow__(x)

    def __rpow__(self, x: int) -> Any:
        return self.value.__rpow__(x)

    def __and__(self, n: int) -> int:
        return self.value.__and__(n)

    def __or__(self, n: int) -> int:
        return self.value.__or__(n)

    def __xor__(self, n: int) -> int:
        return self.value.__xor__(n)

    def __lshift__(self, n: int) -> int:
        return self.value.__lshift__(n)

    def __rshift__(self, n: int) -> int:
        return self.value.__rshift__(n)

    def __rand__(self, n: int) -> int:
        return self.value.__rand__(n)

    def __ror__(self, n: int) -> int:
        return self.value.__ror__(n)

    def __rxor__(self, n: int) -> int:
        return self.value.__rxor__(n)

    def __rlshift__(self, n: int) -> int:
        return self.value.__rlshift__(n)

    def __rrshift__(self, n: int) -> int:
        return self.value.__rrshift__(n)

    def __neg__(self) -> int:
        return self.value.__neg__()

    def __pos__(self) -> int:
        return self.value.__pos__()

    def __invert__(self) -> int:
        return self.value.__invert__()

    def __round__(self, ndigits: Optional[int] = ...) -> int:
        return self.value.__round__(ndigits)

    def __eq__(self, x: object) -> bool:
        return self.value.__eq__(x)

    def __ne__(self, x: object) -> bool:
        return self.value.__ne__(x)

    def __lt__(self, x: int) -> bool:
        return self.value.__lt__(x)

    def __le__(self, x: int) -> bool:
        return self.value.__le__(x)

    def __gt__(self, x: int) -> bool:
        return self.value.__gt__(x)

    def __ge__(self, x: int) -> bool:
        return self.value.__ge__(x)

    def __str__(self) -> str:
        return self.value.__str__()

    def __float__(self) -> float:
        return self.value.__float__()

    def __int__(self) -> int:
        return self.value.__int__()

    def __abs__(self) -> int:
        return self.value.__abs__()

    def __setattr__(self, name: str, value: Any) -> None:
        if name is 'value':
            object.__setattr__(self, name, value)
        else:
            self.value.__setattr__(name, value)

    def __repr__(self) -> str:
        return self.value.__repr__()

    def __format__(self, format_spec: str) -> str:
        return self.value.__format__(format_spec)

    def __getattribute__(self, name: str) -> Any:
        if name is 'value':
            return object.__getattribute__(self, name)
        elif name is '__class__':
            return MutableNum
        else:
            return self.value.__getattribute__(name)

