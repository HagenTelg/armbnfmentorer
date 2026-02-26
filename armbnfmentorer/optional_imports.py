"""
This module provides a class for optional imports. It allows to import a module and check if it is available, without raising an ImportError if it is not. 

Usage
-----
either use one of the predefined optional imports, e.g. statsmodels, timezonefinder, cartopy, etc. or create your own optional import like this:
from atmPy.opt_imports import statsmodels

or use the OptionalImport class directly:
from atmPy.opt_imports import OptionalImport
matplotlib = OptionalImport('matplotlib', submodules=['pyplot', 'colors']) 
"""
class OptionalImport:
    def __init__(self, name, submodules = None):
        self.module_available = False
        self.module = None
        self.name = name

        self.submodules = submodules
        
        self._attempt_import()
        self._attempt_import_submods()

    def _attempt_import_submods(self):
        if (not isinstance(self.submodules, type(None))) and self.module_available:
            submodules = self.submodules
            
            if not isinstance(submodules, list):
                submodules = [submodules,]
                
            for mod in submodules:
                __import__(f'{self.name}.{mod}')
            

    def _attempt_import(self):
        try:
            self.module = __import__(self.name)
            self.module_available = True
        except ImportError:
            self.module_available = False

    def __getattr__(self, item):
        if not self.module_available:
            raise ImportError(f"{self.name} is required for this feature. Please install it to use this functionality.")
        return getattr(self.module, item)



matplotlib = OptionalImport('matplotlib', submodules='pyplot') #The following will replace the import of plt: from atmPy.opt_imports import matplotlib as mpl; mpl.pyplot.plot(...)
IPython = OptionalImport('IPython', submodules='display')
PIL = OptionalImport('PIL', submodules='Image')

