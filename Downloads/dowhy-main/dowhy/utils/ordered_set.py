class OrderedSet:
    """
    Python class for ordered set.
    Code taken from https://github.com/buyalsky/ordered-hash-set/tree/5198b23e01faeac3f5398ab2c08cb013d14b3702.
    """

    def __init__(self, elements=None):
        self._set = {}
        self._start = None
        self._end = None
        if elements is not None:
            for element in elements:
                self.add(element)

    def add(self, element):
        """
        Function to add an element to do set if it does not exit.

        :param element: element to be added.
        """
        if self._start is None:
            self._start = element

        if element not in self._set.keys():
            self._set[element] = None
            if len(self._set) > 1:
                self._set[self._end] = element
            self._end = element

    def get_all(self):
        """
        Function to return list of all elements in the set.

        :returns: List of all items in the set.
        """
        return list(self)

    def is_empty(self):
        """
        Function to determine if the set is empty or not.

        :returns: ``True`` if the set is empty, ``False`` otherwise.
        """
        return self.__len__() == 0

    def union(self, other_set):
        """
        Function to compute the union of self._set and other_set.

        :param other_set: The set to obtain union with. Can be a list, set or OrderedSet.
        :returns: New OrderedSet representing the set with elements from the OrderedSet object and other_set.
        """
        new_set = OrderedSet()
        for element in self._set:
            new_set.add(element)
        for element in other_set:
            new_set.add(element)
        return new_set

    def intersection(self, other_set):
        """
        Function to compute the intersection of self._set and other_set.

        :param other_set: The set to obtain intersection with. Can be a list, set or OrderedSet.
        :returns: New OrderedSet representing the set with elements common to the OrderedSet object and other_set.
        """
        new_set = OrderedSet()
        for element in self._set:
            if element in other_set:
                new_set.add(element)
        return new_set

    def difference(self, other_set):
        """
        Function to remove elements in self._set which are also present in other_set.

        :param other_set: The set to obtain difference with. Can be a list, set or OrderedSet.
        :returns: New OrderedSet representing the difference of elements in the self._set and other_set.
        """
        new_set = OrderedSet()
        for element in self._set:
            if element not in other_set:
                new_set.add(element)
        return new_set

    def __getitem__(self, index):
        if index >= self.__len__():
            raise IndexError("Index is out of range")
        return list(self)[index]

    def __iter__(self):
        self._iter = self._start
        return self

    def __next__(self):
        element = self._iter
        if not element:
            raise StopIteration
        self._iter = self._set[element]
        return element

    def __len__(self):
        return len(self._set)

    def __str__(self):
        elements = [str(i) for i in self]
        string = "OrderedSet(" + ",".join(elements) + ")"
        return string

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return self._set == other._set
