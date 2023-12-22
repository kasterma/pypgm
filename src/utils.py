def flatten(ls: list) -> list:
    """Remove (where possible) one level of list nesting

    >>> flatten([[1,2,3], 4, [5,6]])
    [1, 2, 3, 4, 5, 6]
    """
    rv = []
    for l in ls:  # noqa: E741
        if isinstance(l, list) or isinstance(l, tuple):
            rv.extend(l)
        else:
            rv.append(l)
    return rv
