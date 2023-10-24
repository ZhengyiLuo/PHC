__all__ = ['flags', 'summation']

class Flags(object):
    def __init__(self, *items):
        for key,val in zip(items[:-1], items[1:]):
            setattr(self,key,val)

flags = Flags('debug', False)
