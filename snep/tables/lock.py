import threading
from functools import wraps
'''
The metaclass LockAllFunctionsMeta is used to automatically control access
to the HDF5 file for all the tables classes. This is because the tables
file is accessed both in the main thread and a reader thread used by the
multiprocessing system. Unless the tables are locked we can't guarantee
a read-write collision won't happen.
'''

lock = threading.RLock()

def tablelockerdecorator(func):
    @wraps(func)
    def lockedfunction(self, *args, **kwargs):
        lock.acquire(blocking=True)
        try:
            return func(self, *args, **kwargs)
        finally:
            lock.release()

    return lockedfunction

class LockAllFunctionsMeta(type):
    def __new__(cls, name, bases, local):
        import types
        for attr in local:
            value = local[attr]
            if isinstance(value,types.FunctionType):
                local[attr] = tablelockerdecorator(value)
        return type.__new__(cls, name, bases, local)

