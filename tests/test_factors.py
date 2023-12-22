from factors import RandomVariable, Scope, Factor
import pytest
from hypothesis import given, strategies as st
import string
from itertools import chain, combinations

random_variable = st.builds(
    RandomVariable,
    st.text(alphabet=string.ascii_uppercase, min_size=1),
    st.lists(st.integers()),
)


def nonempty_sublists(l):
    return list(chain.from_iterable(combinations(l, k) for k in range(1, len(l) + 1)))


@st.composite
def scope(draw, base_scope_list_key="key"):
    base_scope_list = draw(
        st.shared(
            st.lists(
                random_variable, min_size=1, max_size=10, unique_by=lambda rv: rv.name
            ),
            key=base_scope_list_key,
        )
    )
    scope = Scope(draw(st.sampled_from(nonempty_sublists(base_scope_list))))
    return scope


def test_random_variable():
    rv = RandomVariable("X", [1, 2, 3])
    assert len(rv) == 3


@given(random_variable)
def test_random_random_variable(r):
    assert r == r


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
    assert s == t

    assert s | t == s
    assert s | t == t
    assert s | u == u
    assert s | u != s

    assert s & t == t
    assert s & u == s


@given(
    scope(base_scope_list_key="tsap"),
    scope(base_scope_list_key="tsap"),
    scope(base_scope_list_key="tsap"),
)
def test_scope_associative_props(s, t, u):
    assert s | (t | u) == (s | t) | u
    assert s & (t & u) == (s & t) & u
    assert (s | t).equiv(t | s)
    assert (s & t).equiv(t & s)


def test_factor():
    x = RandomVariable("X", [1, 2, 3])
    y = RandomVariable("Y", [2, 3, 4])
    s = Scope(x, y)
    f = Factor(s, range(9))

    with pytest.raises(AssertionError):
        Factor(s, range(8))

    f.as_dataframe()

    I = RandomVariable("I", [0, 1])
    D = RandomVariable("D", [0, 1])
    G = RandomVariable("G", [1, 2, 3])
    jd_scope = Scope(I, D, G)
    values = [
        0.126,
        0.168,
        0.126,
        0.009,
        0.045,
        0.126,
        0.252,
        0.0224,
        0.0056,
        0.06,
        0.036,
        0.024,
    ]
    assert sum(values) == pytest.approx(1.0)
    jd_idg = Factor(jd_scope, values)

    id_unnorm = Factor(
        Scope(I, D), [0.126, 0.009, 0.252, 0.06]
    )  # jd_idg restricted to G = 1, not normalized

    cpd_G = Factor(
        jd_scope, [0.3, 0.4, 0.3, 0.05, 0.25, 0.7, 0.9, 0.08, 0.02, 0.5, 0.3, 0.2]
    )  # uncertain about order

    A = RandomVariable("A", [0, 1])
    B = RandomVariable("B", [0, 1])
    phi = Factor(Scope(A, B), [30, 5, 1, 10])

    Aa = RandomVariable("Aa", [1, 2, 3])
    F_1 = Factor(Scope(Aa, B), [0.5, 0.8, 0.1, 0, 0.3, 0.9])
    C = RandomVariable("C", [1, 2])
    F_2 = Factor(Scope(B, C), [0.5, 0.7, 0.1, 0.2])
    F_3 = Factor(
        Scope(Aa, B, C),
        [0.25, 0.35, 0.08, 0.16, 0.05, 0.07, 0, 0, 0.15, 0.21, 0.09, 0.18],
    )
