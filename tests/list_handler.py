import logging
import inspect

class ListHandler(logging.Handler):
    """Simple logging handler that stores all records to a list"""
    def __init__(self):
        super().__init__()
        self.records = []
        self.emit = self.records.append

    def add_to_object_module(self, obj):
        """Given an object, will add itself to the logger of the module in which the object was defined
        
        uses `inspect.getmodule` to find the module
        """
        module = inspect.getmodule(obj)
        logging.getLogger(module.__name__).addHandler(self)
