"""
Additional fallback mechanisms for the Negative Space Imaging Project.

This module provides a centralized location for importing all
external dependencies with appropriate fallbacks, making the project
more robust across different environments.

Usage:
    from src.utils.fallbacks import np, plt, cv2, o3d, web3
"""

import sys
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#-----------------------------------------------------------------------------
# NumPy - Essential, but provide informative error if missing
#-----------------------------------------------------------------------------
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    logger.error("""
    NumPy is required but not installed. 
    Please install it with: pip install numpy
    """)
    # Create minimal numpy-like functionality for basic operations
    class MinimalNumPy:
        def __init__(self):
            self.array = lambda x: x
            self.zeros = lambda shape: [0] * (shape[0] if isinstance(shape, tuple) else shape)
            self.ones = lambda shape: [1] * (shape[0] if isinstance(shape, tuple) else shape)
            
    np = MinimalNumPy()
    NUMPY_AVAILABLE = False

#-----------------------------------------------------------------------------
# Matplotlib - Provide dummy visualization functions if not available
#-----------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    logger.warning("Matplotlib not available. Visualization functions will be limited.")
    
    # Create a minimal matplotlib-like interface
    class DummyFigure:
        def __init__(self, *args, **kwargs):
            pass
        
        def add_subplot(self, *args, **kwargs):
            return DummyAxes()
        
    class DummyAxes:
        def plot(self, *args, **kwargs):
            logger.warning("Plot operation skipped (matplotlib not available)")
            return []
            
        def scatter(self, *args, **kwargs):
            logger.warning("Scatter operation skipped (matplotlib not available)")
            return []
            
        def bar(self, *args, **kwargs):
            logger.warning("Bar plot operation skipped (matplotlib not available)")
            return []
            
        def set_title(self, *args, **kwargs):
            pass
            
        def set_xlabel(self, *args, **kwargs):
            pass
            
        def set_ylabel(self, *args, **kwargs):
            pass
            
        def set_xlim(self, *args, **kwargs):
            pass
            
        def set_ylim(self, *args, **kwargs):
            pass
            
        def grid(self, *args, **kwargs):
            pass
            
        def legend(self, *args, **kwargs):
            pass
    
    # Create a minimal pyplot-like module
    class DummyPyPlot:
        def figure(self, *args, **kwargs):
            logger.warning("Figure creation skipped (matplotlib not available)")
            return DummyFigure()
            
        def subplot(self, *args, **kwargs):
            return DummyAxes()
            
        def plot(self, *args, **kwargs):
            logger.warning("Plot operation skipped (matplotlib not available)")
            return []
            
        def scatter(self, *args, **kwargs):
            logger.warning("Scatter operation skipped (matplotlib not available)")
            return []
            
        def bar(self, *args, **kwargs):
            logger.warning("Bar plot operation skipped (matplotlib not available)")
            return []
            
        def title(self, *args, **kwargs):
            pass
            
        def xlabel(self, *args, **kwargs):
            pass
            
        def ylabel(self, *args, **kwargs):
            pass
            
        def xlim(self, *args, **kwargs):
            pass
            
        def ylim(self, *args, **kwargs):
            pass
            
        def grid(self, *args, **kwargs):
            pass
            
        def legend(self, *args, **kwargs):
            pass
            
        def close(self, *args, **kwargs):
            pass
            
        def savefig(self, *args, **kwargs):
            logger.warning(f"Figure save skipped (matplotlib not available)")
            
        def tight_layout(self, *args, **kwargs):
            pass
    
    plt = DummyPyPlot()
    MATPLOTLIB_AVAILABLE = False

#-----------------------------------------------------------------------------
# OpenCV - Provide text-based image descriptions if not available
#-----------------------------------------------------------------------------
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    logger.warning("OpenCV not available. Image processing functions will be limited.")
    
    # Create a minimal OpenCV-like interface for basic operations
    class DummyOpenCV:
        def __init__(self):
            self.IMREAD_COLOR = 1
            self.IMREAD_GRAYSCALE = 0
            
        def imread(self, filepath, flags=None):
            logger.warning(f"Image read skipped: {filepath} (OpenCV not available)")
            # Return a tiny dummy image (3x3 grayscale)
            if NUMPY_AVAILABLE:
                return np.zeros((3, 3), dtype=np.uint8)
            return [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            
        def imwrite(self, filepath, img):
            logger.warning(f"Image write skipped: {filepath} (OpenCV not available)")
            return False
            
        def resize(self, img, size):
            logger.warning("Image resize skipped (OpenCV not available)")
            return img
            
        def cvtColor(self, img, code):
            logger.warning("Color conversion skipped (OpenCV not available)")
            return img
    
    cv2 = DummyOpenCV()
    OPENCV_AVAILABLE = False

#-----------------------------------------------------------------------------
# Open3D - Already handled in open3d_support.py, but reexported here
#-----------------------------------------------------------------------------
from .open3d_support import o3d, OPEN3D_AVAILABLE

#-----------------------------------------------------------------------------
# Web3 - Provide simulated blockchain operations if not available
#-----------------------------------------------------------------------------
try:
    import web3
    from web3 import Web3
    WEB3_AVAILABLE = True
except ImportError:
    logger.warning("Web3 not available. Blockchain functions will be simulated.")
    
    # Create minimal Web3-like functionality for simulation
    class DummyWeb3:
        def __init__(self):
            self.eth = DummyEth()
            
        def toChecksumAddress(self, address):
            return address
            
        def keccak(self, text=None, hexstr=None):
            import hashlib
            if text:
                return hashlib.sha256(text.encode()).digest()
            elif hexstr:
                return hashlib.sha256(bytes.fromhex(hexstr.replace('0x', ''))).digest()
            return hashlib.sha256(b'').digest()
    
    class DummyEth:
        def __init__(self):
            self.accounts = ["0x0000000000000000000000000000000000000001"]
            self.default_account = self.accounts[0]
            self.contract = lambda abi, address: DummyContract()
    
    class DummyContract:
        def __init__(self):
            pass
            
        def functions(self):
            return self
            
        def call(self):
            return None
            
        def transact(self, transaction=None):
            return "0x0000000000000000000000000000000000000000000000000000000000000000"
    
    web3 = DummyWeb3()
    Web3 = DummyWeb3
    WEB3_AVAILABLE = False

#-----------------------------------------------------------------------------
# Solidity Compiler - For smart contract compilation
#-----------------------------------------------------------------------------
try:
    import solcx
    SOLCX_AVAILABLE = True
except ImportError:
    logger.warning("solcx not available. Smart contract compilation will be simulated.")
    
    # Create minimal solcx-like functionality for simulation
    class DummySolcx:
        def compile_source(self, source, output_values=None):
            return {
                'dummy_contract': {
                    'abi': [],
                    'bin': '0x0'
                }
            }
            
        def install_solc(self, version):
            pass
    
    solcx = DummySolcx()
    SOLCX_AVAILABLE = False

#-----------------------------------------------------------------------------
# Pandas - For temporal analysis
#-----------------------------------------------------------------------------
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    logger.warning("Pandas not available. Temporal analysis will be limited.")
    
    # Create minimal pandas-like functionality
    class DummySeries:
        def __init__(self, data=None, index=None):
            self.data = data or []
            self.index = index or list(range(len(self.data)))
            
        def __getitem__(self, key):
            if isinstance(key, int):
                return self.data[key]
            return self
    
    class DummyDataFrame:
        def __init__(self, data=None, columns=None, index=None):
            self.data = data or {}
            self.columns = columns or (list(data.keys()) if isinstance(data, dict) else [])
            self.index = index or list(range(len(next(iter(data.values())) if data else [])))
            
        def __getitem__(self, key):
            if key in self.data:
                return DummySeries(self.data[key], self.index)
            return DummySeries()
            
        def to_csv(self, path):
            logger.warning(f"CSV export skipped: {path} (Pandas not available)")
    
    class DummyPandas:
        def DataFrame(self, data=None, columns=None, index=None):
            return DummyDataFrame(data, columns, index)
            
        def Series(self, data=None, index=None):
            return DummySeries(data, index)
            
        def read_csv(self, filepath):
            logger.warning(f"CSV read skipped: {filepath} (Pandas not available)")
            return DummyDataFrame()
    
    pd = DummyPandas()
    PANDAS_AVAILABLE = False

#-----------------------------------------------------------------------------
# Crypto modules - For blockchain operations
#-----------------------------------------------------------------------------
try:
    import hashlib
    HASHLIB_AVAILABLE = True
except ImportError:
    logger.warning("hashlib not available. Using simple hash functions.")
    
    # Create minimal hash functionality
    class DummyHashLib:
        def sha256(self, data):
            return DummyHash(data)
            
        def sha512(self, data):
            return DummyHash(data)
            
        def md5(self, data):
            return DummyHash(data)
    
    class DummyHash:
        def __init__(self, data):
            self.data = data
            
        def hexdigest(self):
            # Create a simple deterministic hash
            if hasattr(self.data, 'decode'):
                try:
                    s = self.data.decode('utf-8')
                except:
                    s = str(self.data)
            else:
                s = str(self.data)
            
            value = 0
            for char in s:
                value = (value * 31 + ord(char)) & 0xFFFFFFFF
            
            return format(value, '064x')  # 64-char hex string
    
    hashlib = DummyHashLib()
    HASHLIB_AVAILABLE = False

# Export all modules
__all__ = [
    'np', 'plt', 'cv2', 'o3d', 'web3', 'Web3', 'pd', 'hashlib', 'solcx',
    'NUMPY_AVAILABLE', 'MATPLOTLIB_AVAILABLE', 'OPENCV_AVAILABLE',
    'OPEN3D_AVAILABLE', 'WEB3_AVAILABLE', 'PANDAS_AVAILABLE',
    'HASHLIB_AVAILABLE', 'SOLCX_AVAILABLE'
]
