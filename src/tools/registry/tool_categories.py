"""
Tool Category Definitions and Utilities.

Provides category management and filtering for the tool registry.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from ..definitions.base_tool import ToolCategory


@dataclass
class CategoryInfo:
    """Information about a tool category."""
    category: ToolCategory
    display_name: str
    description: str
    icon: str
    priority: int


CATEGORY_INFO: Dict[ToolCategory, CategoryInfo] = {
    ToolCategory.IMAGING_CORE: CategoryInfo(
        category=ToolCategory.IMAGING_CORE,
        display_name="Core Imaging",
        description="Essential image analysis and processing tools",
        icon="🖼️",
        priority=1
    ),
    ToolCategory.IMAGING_ADVANCED: CategoryInfo(
        category=ToolCategory.IMAGING_ADVANCED,
        display_name="Advanced Imaging",
        description="Specialized image analysis with ML capabilities",
        icon="🔬",
        priority=2
    ),
    ToolCategory.DATABASE: CategoryInfo(
        category=ToolCategory.DATABASE,
        display_name="Database",
        description="Data storage, retrieval, and query operations",
        icon="🗄️",
        priority=3
    ),
    ToolCategory.SECURITY: CategoryInfo(
        category=ToolCategory.SECURITY,
        display_name="Security",
        description="Authentication, encryption, and audit tools",
        icon="🔒",
        priority=4
    ),
    ToolCategory.EXPORT: CategoryInfo(
        category=ToolCategory.EXPORT,
        display_name="Export",
        description="Report generation and format conversion",
        icon="📤",
        priority=5
    ),
    ToolCategory.ML_INFERENCE: CategoryInfo(
        category=ToolCategory.ML_INFERENCE,
        display_name="ML Inference",
        description="Machine learning model predictions",
        icon="🧠",
        priority=6
    ),
    ToolCategory.SPECIALIZED_MEDICAL: CategoryInfo(
        category=ToolCategory.SPECIALIZED_MEDICAL,
        display_name="Medical Imaging",
        description="DICOM and medical format support",
        icon="🏥",
        priority=7
    ),
    ToolCategory.SPECIALIZED_ASTRO: CategoryInfo(
        category=ToolCategory.SPECIALIZED_ASTRO,
        display_name="Astronomical",
        description="FITS and astronomical data processing",
        icon="🔭",
        priority=8
    ),
    ToolCategory.HPC: CategoryInfo(
        category=ToolCategory.HPC,
        display_name="HPC",
        description="High-performance computing operations",
        icon="⚡",
        priority=9
    ),
    ToolCategory.ADMIN: CategoryInfo(
        category=ToolCategory.ADMIN,
        display_name="Administration",
        description="System administration and configuration",
        icon="⚙️",
        priority=10
    ),
    ToolCategory.UTILITY: CategoryInfo(
        category=ToolCategory.UTILITY,
        display_name="Utility",
        description="General purpose utility tools",
        icon="🔧",
        priority=11
    ),
}


def get_category_info(category: ToolCategory) -> CategoryInfo:
    """Get detailed information about a category."""
    return CATEGORY_INFO.get(category, CategoryInfo(
        category=category,
        display_name=category.name.replace("_", " ").title(),
        description="No description available",
        icon="📦",
        priority=99
    ))


def get_categories_by_priority() -> List[CategoryInfo]:
    """Get all categories sorted by priority."""
    return sorted(CATEGORY_INFO.values(), key=lambda x: x.priority)


def get_category_by_name(name: str) -> Optional[ToolCategory]:
    """Get category enum from string name."""
    try:
        return ToolCategory[name.upper()]
    except KeyError:
        return None


def format_category_list() -> str:
    """Format category list for display."""
    lines = []
    for info in get_categories_by_priority():
        lines.append(f"{info.icon} {info.display_name}: {info.description}")
    return "\n".join(lines)
