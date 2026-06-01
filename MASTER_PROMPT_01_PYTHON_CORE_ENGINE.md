# 🐍 MASTER PROMPT 01: PYTHON CORE ENGINE
**Week 1 Execution | Nov 11-15, 2025**

**Copy this entire prompt and paste into GitHub Copilot Chat, then request:**
```
"Generate all 6 Python files"
```

---

## DETAILED PROMPT FOR COPILOT

```
TASK: Create a comprehensive, production-ready Python module for negative space detection in images.

REQUIREMENTS:
- Create 6 Python files with type hints throughout
- Implement edge detection algorithms (Canny, Sobel)
- Implement contour analysis and filtering
- Implement confidence scoring system
- Generate JSON reports for analysis results
- Include comprehensive docstrings and error handling
- Use Pydantic for data validation

FILES TO GENERATE:

1. src/python/negative_space/__init__.py
   - Package initialization
   - Export main NegativeSpaceAnalyzer class
   - Version info

2. src/python/negative_space/core/analyzer.py (400 lines)
   - Class: NegativeSpaceAnalyzer
   - Methods:
     * __init__(self, config: dict = None) -> None
     * analyze(self, image_path: str) -> AnalysisResult
     * analyze_bytes(self, image_data: bytes) -> AnalysisResult
     * batch_analyze(self, image_paths: List[str]) -> List[AnalysisResult]
   - Uses algorithms and image_utils internally
   - Returns AnalysisResult objects

3. src/python/negative_space/core/algorithms.py (300 lines)
   - Function: detect_edges(image: np.ndarray, method: str = 'canny') -> np.ndarray
   - Function: find_contours(edges: np.ndarray) -> List[np.ndarray]
   - Function: filter_contours(contours: List[np.ndarray], min_area: int = 100) -> List[np.ndarray]
   - Function: calculate_confidence(contour: np.ndarray, image: np.ndarray) -> float
   - Function: extract_bounding_boxes(contours: List[np.ndarray]) -> List[Dict]
   - All with proper type hints and documentation

4. src/python/negative_space/core/models.py (150 lines)
   - Pydantic models for data validation
   - Model: ContourData
   - Model: AnalysisResult
   - Model: ConfigModel
   - Include JSON serialization methods

5. src/python/negative_space/utils/image_utils.py (200 lines)
   - Function: load_image(path: str) -> np.ndarray
   - Function: load_image_from_bytes(data: bytes) -> np.ndarray
   - Function: resize_image(image: np.ndarray, max_size: int = 1024) -> np.ndarray
   - Function: convert_to_grayscale(image: np.ndarray) -> np.ndarray
   - Function: enhance_contrast(image: np.ndarray) -> np.ndarray
   - Function: save_visualization(image: np.ndarray, contours: List, path: str) -> None

6. src/python/negative_space/exceptions.py (50 lines)
   - CustomException classes:
     * NegativeSpaceError (base)
     * ImageLoadError
     * AnalysisError
     * ValidationError

IMPLEMENTATION DETAILS:
- Use OpenCV for image processing
- Use NumPy for numerical operations
- Use scikit-image for advanced image processing
- Use Pydantic for data validation
- Include proper error handling and logging
- Add comprehensive docstrings
- Type hint all function parameters and returns
- Include examples in docstrings

TESTING REQUIREMENTS:
- Each file should have a __name__ == '__main__' block with basic tests
- Create a simple test analysis on sample image
- Print success messages for each component

QUALITY STANDARDS:
- All functions documented with docstrings
- All type hints present
- Error handling for edge cases
- Logging statements for debugging
- JSON serializable output
- PEP 8 compliant

STRUCTURE EXAMPLE:
```python
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
import numpy as np
import cv2
from loggers import logger

class NegativeSpaceAnalyzer:
    """Analyzes images to detect negative space regions."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize analyzer with optional config."""
        self.config = config or {}
        logger.info("NegativeSpaceAnalyzer initialized")

    def analyze(self, image_path: str) -> AnalysisResult:
        """Analyze image at path."""
        # Implementation here
        pass
```

Generate all 6 files now.
```

---

## 📋 EXECUTION CHECKLIST

After Copilot generates the files:

- [ ] All 6 files generated successfully
- [ ] Files saved to correct directories
- [ ] No syntax errors in Python files
- [ ] Imports work: `python -c "from negative_space.core.analyzer import NegativeSpaceAnalyzer"`
- [ ] Basic test runs: `python -c "analyzer = NegativeSpaceAnalyzer(); print('✓ Success')"`
- [ ] Files committed to git

## 📍 FILE STRUCTURE

```
src/python/negative_space/
├── __init__.py
├── core/
│   ├── analyzer.py
│   ├── algorithms.py
│   └── models.py
├── utils/
│   └── image_utils.py
└── exceptions.py
```

## 🚀 NEXT STEPS

1. **Week 1 (Nov 11-15):** Complete Master Prompt 01 ✓ THIS WEEK
2. **Week 2 (Nov 18-22):** Use Master Prompt 02 (Express API)
3. **Weeks 3-4:** Use Master Prompt 03 (React Frontend)
4. **Week 5:** Use Master Prompt 04 (Database & Testing)

---

**Status:** Ready for execution
**Created:** November 8, 2025
**Location:** `PHASE_2_EXECUTION_GUIDE.md` (backup) + This standalone file
**Time to Generate:** 5-10 minutes
