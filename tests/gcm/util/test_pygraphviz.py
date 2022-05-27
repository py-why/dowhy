from pytest import approx

from dowhy.gcm.util.pygraphviz import _calc_arrow_width


def test_calc_arrow_width():
    assert _calc_arrow_width(.4, max_strength=.5) == approx(3.3, abs=0.01)
    assert _calc_arrow_width(.2, max_strength=.5) == approx(1.7, abs=0.01)
    assert _calc_arrow_width(-.2, max_strength=.5) == approx(1.7, abs=0.01)
    assert _calc_arrow_width(.5, max_strength=.5) == approx(4.1, abs=0.01)
    assert _calc_arrow_width(.35, max_strength=.5) == approx(2.9, abs=0.01)
    assert _calc_arrow_width(100, max_strength=101) == approx(4.06, abs=0.01)
