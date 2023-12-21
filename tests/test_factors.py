from factors import RandomVariable, Scope, Factor
import pytest


def test_random_variable():
    rv = RandomVariable("X", [1, 2, 3])
    assert len(rv) == 3


def test_scope():
    s = Scope([RandomVariable("X", [1, 2, 3]), RandomVariable("Y", [1, 2, 3])])
    assert len(s) == 9
    u = Scope(
        [
            RandomVariable("X", [1, 2, 3]),
            RandomVariable("Y", [1, 2, 3]),
            RandomVariable("Z", [1, 2, 3]),
        ]
    )
    assert len(u) == 27
    with pytest.raises(AssertionError):
        Scope([RandomVariable("X", [1, 2, 3]), RandomVariable("X", [1, 2, 3])])

    # Note: rvs passed as separate variables
    t = Scope(RandomVariable("X", [1, 2, 3]), RandomVariable("Y", [1, 2, 3]))
    ss = s
    tt = t
    assert s == t


def test_factor():
    x = RandomVariable("X", [1, 2, 3])
    y = RandomVariable("Y", [2, 3, 4])
    s = Scope(x, y)
    f = Factor(s, range(9))

    with pytest.raises(AssertionError):
        Factor(s, range(8))

    f.as_dataframe()
