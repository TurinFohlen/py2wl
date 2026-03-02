import sys
from ._proxy_base import LibraryProxy
sys.modules[__name__] = LibraryProxy("cProfile")
