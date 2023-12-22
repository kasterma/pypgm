from math import prod, isclose
from utils import flatten
import pandas as pd
from itertools import accumulate


class RandomVariable:
    def __init__(self, name: str, values: list):
        self.name = name
        assert len(values) > 0
        self.values = values

    def __len__(self):
        return len(self.values)

    def __eq__(self, other):
        return (
            isinstance(other, RandomVariable)
            and self.name == other.name
            and self.values == other.values
        )

    def __hash__(self):
        return hash((self.name, tuple(self.values)))

    def __repr__(self):
        return f"RandomVariable({self.name}, {self.values})"

    def __getitem__(self, val):
        return self.values.index(val)


class Scope:
    def __init__(self, *args):
        rvs: list[RandomVariable] = flatten(args)
        assert len(rvs) == len(set(rv.name for rv in rvs))
        self.rvs = rvs
        self.rvs_map = {rv.name: rv for rv in self.rvs}
        self.strides = list(
            reversed(
                list(
                    accumulate(
                        [len(rv) for rv in self.rvs[-1:0:-1]],
                        lambda a, b: a * b,
                        initial=1,
                    )
                )
            )
        )

    def _values(self, idx: int) -> dict:
        r = {}
        for rv, stride in zip(self.rvs, self.strides):
            r[rv.name] = rv.values[idx // stride]
            idx %= stride
        return r

    def __len__(self):
        return prod(len(rv) for rv in self.rvs)

    def __eq__(self, other):
        return isinstance(other, Scope) and self.rvs == other.rvs

    def equiv(self, other):
        """Two scopes are equivalent if they have the same random variables, possibly in different order"""
        return set(self.rvs) == set(other.rvs)

    def __or__(self, other):
        assert isinstance(other, Scope)
        or_rvs = []
        or_rvs.extend(self.rvs)
        for rv in other.rvs:
            if rv not in or_rvs:
                or_rvs.append(rv)
        return Scope(or_rvs)

    def __and__(self, other):
        assert isinstance(other, Scope)
        return Scope([rv for rv in self.rvs if rv in other.rvs])

    def __repr__(self):
        return f"Scope({str([rv.name for rv in self.rvs])})"

    def __getitem__(self, val):
        assert all(k in val.keys() for k in self.rvs_map.keys())
        indexes = [rv[val[rv.name]] for rv in self.rvs]
        return sum(idx * stride for idx, stride in zip(indexes, self.strides))


class Factor:
    def __init__(self, scope: Scope, values: list[float]):
        assert len(values) == len(scope)
        self.scope = scope
        self.values = values

    def as_dataframe(self):
        """Convert to dataframe for easy printing"""
        idx = pd.MultiIndex.from_product(
            [rv.values for rv in self.scope.rvs],
            names=[rv.name for rv in self.scope.rvs],
        )
        return pd.Series(data=self.values, index=idx)

    def __eq__(self, other):
        return (
            isinstance(other, Factor)
            and self.scope == other.scope
            and self.values == other.values
        )

    def approx(self, other):
        return (
            isinstance(other, Factor)
            and self.scope == other.scope
            and all(isclose(v1, v2) for v1, v2 in zip(self.values, other.values))
        )

    def __mul__(self, other):
        assert isinstance(other, Factor)
        mul_scope = self.scope | other.scope
        mul_values = [0] * len(mul_scope)
        for idx in range(len(mul_scope)):
            mul_scope_values = mul_scope._values(idx)
            idx_self = self.scope[mul_scope_values]
            idx_other = other.scope[mul_scope_values]
            mul_values[idx] = self.values[idx_self] * other.values[idx_other]
        return Factor(mul_scope, mul_values)
