from math import prod
from utils import flatten
import pandas as pd


class RandomVariable:
    def __init__(self, name: str, values: list):
        self.name = name
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


class Scope:
    def __init__(self, *args):
        rvs: list[RandomVariable] = flatten(args)
        assert len(rvs) == len(set(rv.name for rv in rvs))
        self.rvs = rvs

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
