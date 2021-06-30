class OrderedSet:
    '''
    Python class for ordered set.
    Code taken from https://github.com/buyalsky/ordered-hash-set/tree/5198b23e01faeac3f5398ab2c08cb013d14b3702.
    '''
    def __init__(self, items=None):
        self._set = {}
        self._start = None
        self._end = None
        if items is not None:
            for item in items:
                self.add(item)
        
    def add(self, item):
        """
        Function to add an item to do set if it does not exit.
        Raises TypeError if the item is not hashable.
        :param item: item to be added.
        :return: Index of the added (or existing item) in the set.
        """
        if not self._start:
            self._start = item
        if item not in self._set.keys():
            self._set[item] = item, [self._end, None]
            if len(self._set) != 1:
                self._set[self._end][1][1] = item
            self._end = item
            return self.__len__() - 1
        else:
            return self.get_all().index(item)

    def get_all(self):
        """
        Function to return list of all items in the set.
        :return: List of all items in the set.
        """
        return list(self)

    def is_empty(self):
        """
        Function to determine if the set is empty or not.
        :return: ``True`` if the set is empty, ``False`` otherwise.
        """
        return self.__len__() == 0

    def intersection(self, *other):
        """
        Function to compute the intersection of the set and others.        
        :param *other: The sets to obtain intersection with. Can be a list, set or OrderedSet.
        :return: New OrderedSet representing the set with elements common to the OrderedSet object and all others.
        """
        new_ordered_set = OrderedSet()

        for element in self:
            for obj in other:
                if element not in obj:
                    break
            else:
                new_ordered_set.add(element)

        return new_ordered_set

    def difference(self, *other):
        """
        Function to remove elements in the set which are also present in others.
        :param *other: The sets to obtain difference with. Can be a list, set or OrderedSet.
        :return: New OrderedSet representing the difference of elements in the set and others.
        """
        new_ordered_set = OrderedSet()

        for element in self:
            for obj in other:
                if element in obj:
                    break
            else:
                new_ordered_set.add(element)

        return new_ordered_set

    def union(self, *other):
        """
        Function to compute the union of set and others.        
        :param *other: The sets to obtain union with. Can be a list, set or OrderedSet.
        :return: New OrderedSet representing the set with elements from the OrderedSet object and all others.
        """
        
        new_ordered_set = OrderedSet()

        for element in self:
            new_ordered_set.add(element)

        for obj in other:
            for element in obj:
                new_ordered_set.add(element)

        return new_ordered_set

    def __getitem__(self, index):
        if index < 0:
            return tuple(i for i in self)[index]

        if self.__len__() >= index:
            IndexError("Index is out of range")

        item = self._start

        for i in range(index):
            item = self._set[self._set[item][1][1]][0]

        return item

    def __iter__(self):
        self._next = self._start
        return self

    def __next__(self):
        item = self._next
        if not item:
            raise StopIteration
        self._next = self._set[self._next][1][1]
        return item

    def __len__(self):
        return len(self._set)

    def __str__(self):
        items = tuple(i for i in self)
        return "OrderedSet("+', '.join(['{}']*(len(items))).format(*items) + ")"

    def __eq__(self, other):
        if not isinstance(self, other.__class__):
            return False

        return self._set == other._set