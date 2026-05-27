def shape(x):
    assert isinstance(x, list), f"bad type for shape: {type(x)}"
    if all(not isinstance(xi, list) for xi in x):
        return [len(x)]
    subshapes = [shape(xi) for xi in x]
    for i, subshape_i in enumerate(subshapes[1:]):
        if subshape_i != subshapes[0]:
            raise ValueError(f"shape mismatch at element {i}: {subshape_i} != {subshapes[0]}")
    return [len(x)] + subshapes[0]