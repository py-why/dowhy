import pytest

from dowhy.utils.ordered_set import OrderedSet


def test_basic_creation_and_iteration():
    s = OrderedSet(["a", "b", "c"])
    assert list(s) == ["a", "b", "c"]


def test_preserves_insertion_order():
    s = OrderedSet(["c", "a", "b"])
    assert list(s) == ["c", "a", "b"]


def test_deduplication_preserves_order():
    s = OrderedSet(["a", "b", "a", "c", "b"])
    assert list(s) == ["a", "b", "c"]


def test_empty_set_iteration():
    s = OrderedSet()
    assert list(s) == []


def test_single_element():
    s = OrderedSet(["x"])
    assert list(s) == ["x"]


def test_len():
    s = OrderedSet(["a", "b", "c"])
    assert len(s) == 3


def test_is_empty():
    assert OrderedSet().is_empty()
    assert not OrderedSet(["a"]).is_empty()


def test_get_all():
    s = OrderedSet(["a", "b", "c"])
    assert s.get_all() == ["a", "b", "c"]


def test_getitem():
    s = OrderedSet(["a", "b", "c"])
    assert s[0] == "a"
    assert s[1] == "b"
    assert s[2] == "c"


def test_getitem_out_of_range():
    s = OrderedSet(["a"])
    with pytest.raises(IndexError):
        _ = s[1]


def test_union():
    s1 = OrderedSet(["a", "b"])
    s2 = OrderedSet(["b", "c"])
    result = s1.union(s2)
    assert list(result) == ["a", "b", "c"]


def test_intersection():
    s1 = OrderedSet(["a", "b", "c"])
    s2 = OrderedSet(["b", "c", "d"])
    result = s1.intersection(s2)
    assert list(result) == ["b", "c"]


def test_difference():
    s1 = OrderedSet(["a", "b", "c"])
    s2 = OrderedSet(["b"])
    result = s1.difference(s2)
    assert list(result) == ["a", "c"]


def test_equality():
    s1 = OrderedSet(["a", "b"])
    s2 = OrderedSet(["a", "b"])
    assert s1 == s2


def test_inequality():
    s1 = OrderedSet(["a", "b"])
    s2 = OrderedSet(["b", "a"])
    assert s1 != s2


def test_str():
    s = OrderedSet(["a", "b"])
    assert str(s) == "OrderedSet(a,b)"


def test_falsy_integer_elements():
    """Regression test: OrderedSet.__next__ used 'if not element' which
    incorrectly raised StopIteration for falsy values such as 0."""
    s = OrderedSet([0, 1, 2])
    assert list(s) == [0, 1, 2]


def test_falsy_integer_zero_only():
    s = OrderedSet([0])
    assert list(s) == [0]


def test_iteration_with_zero_in_middle():
    s = OrderedSet([1, 0, 2])
    assert list(s) == [1, 0, 2]
