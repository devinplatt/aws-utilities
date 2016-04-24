import datetime
import os


def timeit(method):

    def timed(*args, **kw):
        st = datetime.datetime.now()
        result = method(*args, **kw)
        et = datetime.datetime.now()
        print('{} ran in : {}'.format(
              method.__name__, et-st))
        return result

    return timed


def ensure_dirs_exist(paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)