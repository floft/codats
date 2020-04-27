#!/usr/bin/env python3
"""
Test heterogeneous multiple inheritance
"""


class MethodBase:
    def __init__(self):
        print("MethodBase")


class MethodDann(MethodBase):
    def __init__(self):
        super().__init__()
        print("MethodDann")


class MethodDaws(MethodBase):
    def __init__(self):
        super().__init__()
        print("MethodDaws")


class HeterogeneousBase:
    def __init__(self):
        super().__init__()
        print("HeterogeneousBase")


class HeterogeneousDann(HeterogeneousBase, MethodDann):
    pass
    # def __init__(self):
    #     super().__init__()
    #     print("HeterogeneousDann")
    #     print(HeterogeneousDann.__mro__)


class HeterogeneousDaws(HeterogeneousBase, MethodDaws):
    pass
    # def __init__(self):
    #     super().__init__()
    #     print("HeterogeneousDaws")
    #     print(HeterogeneousDaws.__mro__)


if __name__ == "__main__":
    a = HeterogeneousDann()
    # b = HeterogeneousDaws()
