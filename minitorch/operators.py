"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(a: float, b: float):
    return a * b

def id(a: float):
    return a

def add(a: float, b: float):
    return a + b

def neg(a: float):
    return -a

def lt(a: float, b: float):
    return a < b

def eq(a: float, b: float):
    return a == b

def max(a: float, b: float):
    return a if a > b else b

def is_close(a: float, b: float):
    return abs(a - b) < max(1e-5, 1e-8 * max(abs(a), abs(b)))

def sigmoid(x: float):
    return 1 / (1 + exp(-x))

def relu(x: float):
    return x if x >= 0 else 0

def log(x: float):
    return math.log(x)

def exp(x: float):
    return math.exp(x)

def inv(x: float):
    return 1 / x

def log_back(f: float, df: float):
    return df / f

def inv_back(f: float, df: float):
    return -df / (f * f + 1e-8)

def relu_back(f: float, df: float):
    return df if f >= 0 else 0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(func: Callable, iterator: Iterable):
    for obj in iterator:
        yield func(obj)

def zipWith(func: Callable, iterator1: Iterable, iterator2: Iterable):
    it1, it2 = iter(iterator1), iter(iterator2)
    obj1, obj2 = next(it1, None), next(it2, None)
    while (obj1 is not None) and (obj2 is not None):
        yield func(obj1, obj2)
        obj1, obj2 = next(it1, None), next(it2, None)

def reduce(func: Callable, iterator: Iterable, initializer=None):
    it = iter(iterator)
    reduced = initializer if initializer is not None else next(it, None)
    nxt = next(it, None)
    while nxt is not None:
        reduced = func(reduced, nxt)
        nxt = next(it, None)
    return reduced

def negList(lst: list) -> list:
    return list(map(neg, lst))

def addLists(lst1: list, lst2: list):
    return list(zipWith(add, lst1, lst2))

def sum(lst: list):
    return reduce(add, lst, 0)

def prod(lst: list):
    return reduce(mul, lst, 1)
