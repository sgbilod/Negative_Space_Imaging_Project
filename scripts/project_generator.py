#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Universal Project Generator and Manager
Author: Stephen Bilodeau
Date: August 13, 2025

A comprehensive tool for automated project creation, management, and organization
across multiple platforms and interfaces.
"""

import os
import sys
import argparse
import json
import shutil
import datetime
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.expanduser('~/project_manager.log'))
    ]
)

logger = logging.getLogger('project_manager')

# Configuration
DEFAULT_PROJECTS_DIR = os.path.expanduser('~/Projects')
DEFAULT_TEMPLATES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'templates'
)
CONFIG_FILE = os.path.expanduser('~/.project_manager_config.json')
INDEX_FILE = os.path.join(DEFAULT_PROJECTS_DIR, 'project_index.json')
AUTHOR = "Stephen Bilodeau"
GIT_USERNAME = "StephenBilodeau"
GIT_EMAIL = "stephenbilodeau@example.com"  # Replace with your actual email

# Template structure
TEMPLATE_STRUCTURE = {
    "default": {
        "directories": [
            "src",
            "tests",
            "docs",
            "configs",
            ".github/workflows"
        ],
        "files": {
            "README.md": "# {project_name}\n\n{description}\n\n## Overview\n\nProject created on {date}.\n\n## Author\n\n{author}",
            "LICENSE": "Copyright (c) {year} {author}. All rights reserved.",
            ".gitignore": "# Project-specific files\n*.log\n*.tmp\n\n# Environment\n.env\n.venv\nenv/\nvenv/\nENV/\n\n# Python\n__pycache__/\n*.py[cod]\n*$py.class\n*.so\n.Python\nbuild/\ndevelop-eggs/\ndist/\ndownloads/\neggs/\n.eggs/\nlib/\nlib64/\nparts/\nsdist/\nvar/\nwheels/\n*.egg-info/\n.installed.cfg\n*.egg\n\n# Node.js\nnode_modules/\nnpm-debug.log\nyarn-error.log\n.npm\n.yarn-integrity\n\n# IDEs and editors\n.idea/\n.vscode/\n*.swp\n*.swo\n*~\n.DS_Store",
            "CONTRIBUTING.md": "# Contributing to {project_name}\n\nThank you for considering contributing to this project!\n\n## Guidelines\n\n1. Create a feature branch\n2. Make your changes\n3. Submit a pull request\n\n## Code Standards\n\nPlease follow the existing code style and include tests for new features.\n\n## Author\n\n{author}",
            ".github/workflows/ci.yml": "name: CI\n\non:\n  push:\n    branches: [ main ]\n  pull_request:\n    branches: [ main ]\n\njobs:\n  build:\n    runs-on: ubuntu-latest\n    steps:\n    - uses: actions/checkout@v2\n    - name: Build and Test\n      run: echo \"Replace with actual build commands\"\n"
        },
        "dependencies": [],
        "git_init": True,
        "vs_code": {
            "extensions": [
                "GitHub.copilot",
                "GitHub.vscode-pull-request-github"
            ],
            "settings": {
                "editor.formatOnSave": True,
                "editor.codeActionsOnSave": {
                    "source.fixAll": True
                }
            }
        }
    },
    "python": {
        "extends": "default",
        "directories": [
            "src/{project_name_snake}",
            "tests",
            "docs",
            "data",
            "notebooks",
            ".github/workflows"
        ],
        "files": {
            "src/{project_name_snake}/__init__.py": "\"\"\"The {project_name} package.\"\"\"\n\n__version__ = '0.1.0'\n__author__ = '{author}'",
            "src/{project_name_snake}/main.py": "#!/usr/bin/env python\n# -*- coding: utf-8 -*-\n\n\"\"\"\n{project_name}\n\n{description}\n\nAuthor: {author}\nDate: {date}\n\"\"\"\n\nimport argparse\nimport sys\n\n\ndef main():\n    \"\"\"Main entry point for the application.\"\"\"\n    parser = argparse.ArgumentParser(description='{description}')\n    parser.add_argument('--version', action='store_true', help='Show version')\n    args = parser.parse_args()\n    \n    if args.version:\n        from . import __version__\n        print(f'{project_name} version {__version__}')\n        return 0\n    \n    # Your code here\n    print('Hello from {project_name}!')\n    return 0\n\n\nif __name__ == '__main__':\n    sys.exit(main())\n",
            "setup.py": "#!/usr/bin/env python\n# -*- coding: utf-8 -*-\n\nfrom setuptools import setup, find_packages\n\nwith open('README.md') as readme_file:\n    readme = readme_file.read()\n\nrequirements = [\n    # Add your project requirements here\n]\n\nsetup(\n    name='{project_name_snake}',\n    version='0.1.0',\n    description='{description}',\n    long_description=readme,\n    long_description_content_type='text/markdown',\n    author='{author}',\n    author_email='{email}',\n    url='https://github.com/{git_username}/{project_name_snake}',\n    packages=find_packages(where='src'),\n    package_dir={{'': 'src'}},\n    include_package_data=True,\n    install_requires=requirements,\n    license='Proprietary',\n    zip_safe=False,\n    keywords='{project_name_snake}',\n    classifiers=[\n        'Development Status :: 3 - Alpha',\n        'Intended Audience :: Developers',\n        'Natural Language :: English',\n        'Programming Language :: Python :: 3',\n        'Programming Language :: Python :: 3.9',\n        'Programming Language :: Python :: 3.10',\n    ],\n    entry_points={{\n        'console_scripts': [\n            '{project_name_snake}={project_name_snake}.main:main',\n        ],\n    }},\n)\n",
            "requirements.txt": "# Development requirements\npytest>=7.0.0\npytest-cov>=4.1.0\nblack>=23.0.0\nflake8>=6.0.0\nmypy>=1.0.0\n",
            "requirements-dev.txt": "-r requirements.txt\ntwine>=4.0.0\nbuild>=1.0.0\npytest>=7.0.0\npytest-cov>=4.1.0\nblack>=23.0.0\nflake8>=6.0.0\nmypy>=1.0.0\n",
            "pyproject.toml": "[build-system]\nrequires = [\"setuptools>=42\", \"wheel\"]\nbuild-backend = \"setuptools.build_meta\"\n\n[tool.black]\nline-length = 88\ntarget-version = ['py39']\ninclude = '\\.pyi?$'\n\n[tool.isort]\nprofile = \"black\"\n\n[tool.mypy]\npython_version = \"3.9\"\nwarn_return_any = true\nwarn_unused_configs = true\ndisallow_untyped_defs = true\ndisallow_incomplete_defs = true\n",
            ".github/workflows/python-package.yml": "name: Python Package\n\non:\n  push:\n    branches: [ main ]\n  pull_request:\n    branches: [ main ]\n\njobs:\n  build:\n    runs-on: ubuntu-latest\n    strategy:\n      matrix:\n        python-version: [3.9, 3.10, 3.11]\n\n    steps:\n    - uses: actions/checkout@v3\n    - name: Set up Python ${{ matrix.python-version }}\n      uses: actions/setup-python@v4\n      with:\n        python-version: ${{ matrix.python-version }}\n    - name: Install dependencies\n      run: |\n        python -m pip install --upgrade pip\n        pip install -r requirements-dev.txt\n        pip install -e .\n    - name: Lint with flake8\n      run: |\n        flake8 src tests\n    - name: Format check with black\n      run: |\n        black --check src tests\n    - name: Type check with mypy\n      run: |\n        mypy src\n    - name: Test with pytest\n      run: |\n        pytest tests/ --cov=src --cov-report=xml\n",
            "tests/__init__.py": "",
            "tests/test_main.py": "#!/usr/bin/env python\n# -*- coding: utf-8 -*-\n\n\"\"\"\nTests for `{project_name_snake}` package.\n\"\"\"\n\nimport pytest\nfrom {project_name_snake} import main\n\n\ndef test_main_function():\n    \"\"\"Test the main function.\"\"\"\n    assert main.main() == 0\n"
        },
        "dependencies": [
            "python>=3.9",
            "pip",
            "setuptools",
            "wheel",
            "pytest",
            "black",
            "flake8",
            "mypy"
        ],
        "vs_code": {
            "extensions": [
                "ms-python.python",
                "ms-python.vscode-pylance",
                "ms-python.flake8",
                "ms-python.black-formatter",
                "GitHub.copilot",
                "njpwerner.autodocstring"
            ],
            "settings": {
                "editor.formatOnSave": True,
                "python.linting.enabled": True,
                "python.linting.flake8Enabled": True,
                "python.formatting.provider": "black",
                "python.analysis.typeCheckingMode": "basic",
                "editor.codeActionsOnSave": {
                    "source.organizeImports": True
                }
            }
        }
    },
    "web": {
        "extends": "default",
        "directories": [
            "src",
            "public",
            "assets",
            "styles",
            "scripts",
            "components",
            "tests",
            "docs",
            ".github/workflows"
        ],
        "files": {
            "package.json": "{\n  \"name\": \"{project_name_snake}\",\n  \"version\": \"0.1.0\",\n  \"description\": \"{description}\",\n  \"main\": \"index.js\",\n  \"scripts\": {\n    \"start\": \"echo \\\"Add your start script here\\\"\",\n    \"build\": \"echo \\\"Add your build script here\\\"\",\n    \"test\": \"echo \\\"Add your test script here\\\"\",\n    \"lint\": \"eslint src\"\n  },\n  \"author\": \"{author}\",\n  \"license\": \"UNLICENSED\",\n  \"private\": true\n}\n",
            ".eslintrc.json": "{\n  \"env\": {\n    \"browser\": true,\n    \"es2021\": true,\n    \"node\": true\n  },\n  \"extends\": [\n    \"eslint:recommended\"\n  ],\n  \"parserOptions\": {\n    \"ecmaVersion\": 12,\n    \"sourceType\": \"module\"\n  },\n  \"rules\": {\n    \"indent\": [\"error\", 2],\n    \"linebreak-style\": [\"error\", \"unix\"],\n    \"quotes\": [\"error\", \"single\"],\n    \"semi\": [\"error\", \"always\"]\n  }\n}\n",
            "src/index.js": "/**\n * {project_name}\n * \n * {description}\n * \n * @author {author}\n * @date {date}\n */\n\nconsole.log('Hello from {project_name}!');\n",
            "public/index.html": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n  <meta charset=\"UTF-8\">\n  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n  <title>{project_name}</title>\n  <link rel=\"stylesheet\" href=\"../styles/main.css\">\n</head>\n<body>\n  <h1>{project_name}</h1>\n  <p>{description}</p>\n  \n  <script src=\"../src/index.js\"></script>\n</body>\n</html>\n",
            "styles/main.css": "/* Main stylesheet for {project_name} */\n\nbody {\n  font-family: Arial, sans-serif;\n  line-height: 1.6;\n  margin: 0;\n  padding: 20px;\n  color: #333;\n}\n\nh1 {\n  color: #2c3e50;\n}\n"
        },
        "dependencies": [
            "node>=14.0.0",
            "npm",
            "eslint"
        ],
        "vs_code": {
            "extensions": [
                "dbaeumer.vscode-eslint",
                "esbenp.prettier-vscode",
                "ritwickdey.LiveServer",
                "GitHub.copilot"
            ],
            "settings": {
                "editor.formatOnSave": True,
                "editor.defaultFormatter": "esbenp.prettier-vscode",
                "prettier.singleQuote": True,
                "prettier.printWidth": 100,
                "eslint.validate": ["javascript", "typescript"],
                "editor.codeActionsOnSave": {
                    "source.fixAll.eslint": True
                }
            }
        }
    },
    "research": {
        "extends": "python",
        "directories": [
            "src/{project_name_snake}",
            "notebooks",
            "data/raw",
            "data/processed",
            "data/external",
            "results/figures",
            "results/tables",
            "papers",
            "presentations",
            "tests",
            "docs"
        ],
        "files": {
            "notebooks/exploration.ipynb": "{\n \"cells\": [\n  {\n   \"cell_type\": \"markdown\",\n   \"metadata\": {},\n   \"source\": [\n    \"# {project_name} - Data Exploration\\n\",\n    \"\\n\",\n    \"{description}\\n\",\n    \"\\n\",\n    \"Author: {author}\\n\",\n    \"Date: {date}\"\n   ]\n  },\n  {\n   \"cell_type\": \"code\",\n   \"execution_count\": null,\n   \"metadata\": {},\n   \"outputs\": [],\n   \"source\": [\n    \"import numpy as np\\n\",\n    \"import pandas as pd\\n\",\n    \"import matplotlib.pyplot as plt\\n\",\n    \"import seaborn as sns\\n\",\n    \"\\n\",\n    \"# Set plotting style\\n\",\n    \"plt.style.use('seaborn-v0_8-whitegrid')\\n\",\n    \"sns.set_context('notebook')\\n\",\n    \"\\n\",\n    \"# Display settings\\n\",\n    \"%matplotlib inline\\n\",\n    \"pd.set_option('display.max_columns', None)\"\n   ]\n  },\n  {\n   \"cell_type\": \"markdown\",\n   \"metadata\": {},\n   \"source\": [\n    \"## Data Loading\\n\",\n    \"\\n\",\n    \"Load and examine the dataset.\"\n   ]\n  },\n  {\n   \"cell_type\": \"code\",\n   \"execution_count\": null,\n   \"metadata\": {},\n   \"outputs\": [],\n   \"source\": [\n    \"# TODO: Load your data\\n\",\n    \"# data = pd.read_csv('../data/raw/your_data.csv')\\n\",\n    \"\\n\",\n    \"# Create sample data for initial exploration\\n\",\n    \"np.random.seed(42)\\n\",\n    \"data = pd.DataFrame({\\n\",\n    \"    'feature1': np.random.normal(0, 1, 100),\\n\",\n    \"    'feature2': np.random.normal(5, 2, 100),\\n\",\n    \"    'target': np.random.randint(0, 2, 100)\\n\",\n    \"})\\n\",\n    \"\\n\",\n    \"data.head()\"\n   ]\n  }\n ],\n \"metadata\": {\n  \"kernelspec\": {\n   \"display_name\": \"Python 3\",\n   \"language\": \"python\",\n   \"name\": \"python3\"\n  },\n  \"language_info\": {\n   \"codemirror_mode\": {\n    \"name\": \"ipython\",\n    \"version\": 3\n   },\n   \"file_extension\": \".py\",\n   \"mimetype\": \"text/x-python\",\n   \"name\": \"python\",\n   \"nbconvert_exporter\": \"python\",\n   \"pygments_lexer\": \"ipython3\",\n   \"version\": \"3.9.0\"\n  }\n },\n \"nbformat\": 4,\n \"nbformat_minor\": 4\n}\n",
            "src/{project_name_snake}/data_processing.py": "#!/usr/bin/env python\n# -*- coding: utf-8 -*-\n\n\"\"\"\nData processing utilities for {project_name}.\n\nAuthor: {author}\nDate: {date}\n\"\"\"\n\nimport os\nimport pandas as pd\nimport numpy as np\nfrom typing import Dict, List, Optional, Union, Any\n\n\ndef load_data(file_path: str) -> pd.DataFrame:\n    \"\"\"Load data from various file formats.\n    \n    Args:\n        file_path: Path to the data file\n        \n    Returns:\n        Loaded data as a pandas DataFrame\n    \"\"\"\n    ext = os.path.splitext(file_path)[1].lower()\n    \n    if ext == '.csv':\n        return pd.read_csv(file_path)\n    elif ext in ['.xls', '.xlsx']:\n        return pd.read_excel(file_path)\n    elif ext == '.json':\n        return pd.read_json(file_path)\n    elif ext == '.parquet':\n        return pd.read_parquet(file_path)\n    else:\n        raise ValueError(f\"Unsupported file format: {ext}\")\n\n\ndef preprocess_data(data: pd.DataFrame) -> pd.DataFrame:\n    \"\"\"Preprocess the data for analysis.\n    \n    Args:\n        data: Raw data as a pandas DataFrame\n        \n    Returns:\n        Preprocessed data\n    \"\"\"\n    # Make a copy to avoid modifying the original data\n    df = data.copy()\n    \n    # Basic preprocessing steps\n    # 1. Handle missing values\n    df = df.dropna()\n    \n    # 2. Convert data types if needed\n    # TODO: Add specific type conversions for your data\n    \n    # 3. Feature engineering\n    # TODO: Add feature engineering steps for your data\n    \n    return df\n",
            "requirements.txt": "# Scientific Python\nnumpy>=1.20.0\npandas>=1.3.0\nscipy>=1.7.0\nscikit-learn>=1.0.0\nmatplotlib>=3.4.0\nseaborn>=0.11.0\njupyter>=1.0.0\nnotebook>=6.4.0\n\n# Data formats\nopenpyxl>=3.0.0\nfastparquet>=0.8.0\npyarrow>=7.0.0\n\n# Development requirements\npytest>=7.0.0\npytest-cov>=4.1.0\nblack>=23.0.0\nflake8>=6.0.0\nmypy>=1.0.0\n"
        },
        "dependencies": [
            "python>=3.9",
            "pip",
            "jupyter",
            "numpy",
            "pandas",
            "matplotlib",
            "seaborn",
            "scikit-learn"
        ],
        "vs_code": {
            "extensions": [
                "ms-python.python",
                "ms-python.vscode-pylance",
                "ms-toolsai.jupyter",
                "GitHub.copilot"
            ]
        }
    },
    "data": {
        "extends": "research",
        "directories": [
            "src/{project_name_snake}",
            "src/{project_name_snake}/models",
            "src/{project_name_snake}/features",
            "src/{project_name_snake}/visualization",
            "notebooks",
            "data/raw",
            "data/processed",
            "data/external",
            "models",
            "reports/figures",
            "reports/tables",
            "tests"
        ],
        "files": {
            "src/{project_name_snake}/features/build_features.py": "#!/usr/bin/env python\n# -*- coding: utf-8 -*-\n\n\"\"\"\nFeature engineering module for {project_name}.\n\nAuthor: {author}\nDate: {date}\n\"\"\"\n\nimport pandas as pd\nimport numpy as np\nfrom typing import Dict, List, Optional, Union, Any\n\n\ndef create_features(df: pd.DataFrame) -> pd.DataFrame:\n    \"\"\"Create features from raw data.\n    \n    Args:\n        df: Input DataFrame\n        \n    Returns:\n        DataFrame with engineered features\n    \"\"\"\n    # Make a copy to avoid modifying the original data\n    result = df.copy()\n    \n    # TODO: Add feature engineering steps specific to your project\n    \n    return result\n\n\ndef select_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:\n    \"\"\"Select relevant features for modeling.\n    \n    Args:\n        df: DataFrame with all features\n        target_col: Name of the target column\n        \n    Returns:\n        DataFrame with selected features\n    \"\"\"\n    # TODO: Implement feature selection logic\n    \n    return df\n",
            "src/{project_name_snake}/models/train_model.py": "#!/usr/bin/env python\n# -*- coding: utf-8 -*-\n\n\"\"\"\nModel training module for {project_name}.\n\nAuthor: {author}\nDate: {date}\n\"\"\"\n\nimport os\nimport pickle\nimport pandas as pd\nimport numpy as np\nfrom typing import Dict, List, Optional, Union, Any, Tuple\n\nfrom sklearn.model_selection import train_test_split, cross_val_score\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score\n\n# Example model import\nfrom sklearn.ensemble import RandomForestClassifier\n\n\ndef prepare_data(\n    df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42\n) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:\n    \"\"\"Prepare data for modeling.\n    \n    Args:\n        df: Input DataFrame\n        target_col: Name of the target column\n        test_size: Proportion of data to use for testing\n        random_state: Random seed for reproducibility\n        \n    Returns:\n        X_train, X_test, y_train, y_test\n    \"\"\"\n    X = df.drop(columns=[target_col])\n    y = df[target_col]\n    \n    return train_test_split(X, y, test_size=test_size, random_state=random_state)\n\n\ndef build_model() -> Pipeline:\n    \"\"\"Build a machine learning model pipeline.\n    \n    Returns:\n        Scikit-learn Pipeline\n    \"\"\"\n    # Example pipeline with preprocessing and model\n    pipeline = Pipeline([\n        ('scaler', StandardScaler()),\n        ('model', RandomForestClassifier(random_state=42))\n    ])\n    \n    return pipeline\n\n\ndef train_and_evaluate(\n    X_train: np.ndarray, \n    X_test: np.ndarray, \n    y_train: np.ndarray, \n    y_test: np.ndarray\n) -> Tuple[Pipeline, Dict[str, float]]:\n    \"\"\"Train and evaluate a model.\n    \n    Args:\n        X_train: Training features\n        X_test: Test features\n        y_train: Training target\n        y_test: Test target\n        \n    Returns:\n        Trained model and performance metrics\n    \"\"\"\n    model = build_model()\n    model.fit(X_train, y_train)\n    \n    # Evaluate\n    y_pred = model.predict(X_test)\n    \n    metrics = {\n        'accuracy': accuracy_score(y_test, y_pred),\n        'precision': precision_score(y_test, y_pred, average='weighted'),\n        'recall': recall_score(y_test, y_pred, average='weighted'),\n        'f1': f1_score(y_test, y_pred, average='weighted')\n    }\n    \n    return model, metrics\n\n\ndef save_model(model: Pipeline, filepath: str) -> None:\n    \"\"\"Save model to disk.\n    \n    Args:\n        model: Trained model\n        filepath: Path to save the model\n    \"\"\"\n    with open(filepath, 'wb') as f:\n        pickle.dump(model, f)\n    \n    print(f\"Model saved to {filepath}\")\n\n\ndef load_model(filepath: str) -> Pipeline:\n    \"\"\"Load model from disk.\n    \n    Args:\n        filepath: Path to the saved model\n        \n    Returns:\n        Loaded model\n    \"\"\"\n    with open(filepath, 'rb') as f:\n        model = pickle.load(f)\n    \n    return model\n",
            "src/{project_name_snake}/visualization/visualize.py": "#!/usr/bin/env python\n# -*- coding: utf-8 -*-\n\n\"\"\"\nVisualization module for {project_name}.\n\nAuthor: {author}\nDate: {date}\n\"\"\"\n\nimport os\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom typing import Dict, List, Optional, Union, Any, Tuple\n\n\ndef configure_plots() -> None:\n    \"\"\"Configure plot appearance.\"\"\"\n    plt.style.use('seaborn-v0_8-whitegrid')\n    sns.set_context('notebook')\n    plt.rcParams['figure.figsize'] = (12, 8)\n    plt.rcParams['savefig.dpi'] = 300\n\n\ndef plot_distributions(\n    df: pd.DataFrame, \n    columns: List[str] = None, \n    save_dir: Optional[str] = None\n) -> None:\n    \"\"\"Plot distributions of features.\n    \n    Args:\n        df: Input DataFrame\n        columns: List of columns to plot (if None, use all numeric columns)\n        save_dir: Directory to save plots (if None, display instead)\n    \"\"\"\n    configure_plots()\n    \n    if columns is None:\n        columns = df.select_dtypes(include=['number']).columns.tolist()\n    \n    for col in columns:\n        plt.figure()\n        sns.histplot(df[col], kde=True)\n        plt.title(f'Distribution of {col}')\n        plt.tight_layout()\n        \n        if save_dir:\n            os.makedirs(save_dir, exist_ok=True)\n            plt.savefig(os.path.join(save_dir, f'dist_{col}.png'))\n            plt.close()\n        else:\n            plt.show()\n\n\ndef plot_correlation_matrix(\n    df: pd.DataFrame, \n    method: str = 'pearson',\n    save_path: Optional[str] = None\n) -> None:\n    \"\"\"Plot correlation matrix of features.\n    \n    Args:\n        df: Input DataFrame\n        method: Correlation method ('pearson', 'spearman', or 'kendall')\n        save_path: Path to save the plot (if None, display instead)\n    \"\"\"\n    configure_plots()\n    \n    # Calculate correlation matrix\n    corr = df.corr(method=method)\n    \n    # Generate mask for the upper triangle\n    mask = np.triu(np.ones_like(corr, dtype=bool))\n    \n    # Set up the figure\n    plt.figure(figsize=(14, 12))\n    \n    # Generate heatmap\n    sns.heatmap(\n        corr, \n        mask=mask, \n        cmap='coolwarm', \n        annot=True, \n        fmt='.2f', \n        square=True, \n        linewidths=0.5, \n        cbar_kws={'shrink': 0.8}\n    )\n    \n    plt.title(f'{method.capitalize()} Correlation Matrix')\n    plt.tight_layout()\n    \n    if save_path:\n        os.makedirs(os.path.dirname(save_path), exist_ok=True)\n        plt.savefig(save_path)\n        plt.close()\n    else:\n        plt.show()\n",
            "requirements.txt": "# Scientific Python\nnumpy>=1.20.0\npandas>=1.3.0\nscipy>=1.7.0\nscikit-learn>=1.0.0\nmatplotlib>=3.4.0\nseaborn>=0.11.0\njupyter>=1.0.0\nnotebook>=6.4.0\n\n# Machine Learning\nscikit-learn>=1.0.0\ntensorflow>=2.8.0\nkeras>=2.8.0\nxgboost>=1.5.0\nlightgbm>=3.3.0\n\n# Data Visualization\nplotly>=5.6.0\nbokeh>=2.4.0\n\n# Data formats\nopenpyxl>=3.0.0\nfastparquet>=0.8.0\npyarrow>=7.0.0\n\n# Development requirements\npytest>=7.0.0\npytest-cov>=4.1.0\nblack>=23.0.0\nflake8>=6.0.0\nmypy>=1.0.0\n",
            "Makefile": "# Makefile for {project_name}\n# Author: {author}\n# Date: {date}\n\n.PHONY: clean data lint format test docs help\n\nhelp:\n\t@echo \"Commands:\"  \n\t@echo \"  clean     : remove build artifacts\"  \n\t@echo \"  data      : process raw data into features\"  \n\t@echo \"  lint      : check style with flake8\"  \n\t@echo \"  format    : format code with black\"  \n\t@echo \"  test      : run tests\"  \n\t@echo \"  docs      : generate documentation\"  \n\nclean:\n\t@echo \"Cleaning directories...\"\n\tfind . -type d -name __pycache__ -exec rm -rf {} +\n\tfind . -type f -name \"*.pyc\" -delete\n\tfind . -type f -name \"*.pyo\" -delete\n\tfind . -type f -name \"*.pyd\" -delete\n\tfind . -type f -name \".coverage\" -delete\n\tfind . -type d -name \"*.egg-info\" -exec rm -rf {} +\n\tfind . -type d -name \"*.egg\" -exec rm -rf {} +\n\tfind . -type d -name \".pytest_cache\" -exec rm -rf {} +\n\trm -rf build/\n\trm -rf dist/\n\trm -rf reports/\n\n\ndata:\n\t@echo \"Processing data...\"\n\tpython src/{project_name_snake}/data_processing.py\n\n\nlint:\n\t@echo \"Linting with flake8...\"\n\tflake8 src tests\n\t\nformat:\n\t@echo \"Formatting with black...\"\n\tblack src tests\n\n\ntest:\n\t@echo \"Running tests...\"\n\tpytest --cov=src tests/\n\n\ndocs:\n\t@echo \"Generating documentation...\"\n\t@echo \"TODO: Add documentation generation command\"\n\n"
        },
        "dependencies": [
            "python>=3.9",
            "numpy",
            "pandas",
            "matplotlib",
            "seaborn",
            "scikit-learn",
            "tensorflow",
            "jupyter"
        ]
    }
}


def load_config() -> Dict[str, Any]:
    """Load configuration from file or create default."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config file: {e}")

    # Default configuration
    config = {
        "projects_dir": DEFAULT_PROJECTS_DIR,
        "templates_dir": DEFAULT_TEMPLATES_DIR,
        "author": AUTHOR,
        "git_username": GIT_USERNAME,
        "git_email": GIT_EMAIL,
        "vs_code_enabled": True,
        "git_enabled": True,
        "auto_open_vs_code": True,
        "index_file": INDEX_FILE
    }

    # Save default config
    save_config(config)
    return config


def save_config(config: Dict[str, Any]) -> None:
    """Save configuration to file."""
    try:
        os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save config file: {e}")


def load_project_index() -> Dict[str, Any]:
    """Load project index from file or create default."""
    if os.path.exists(INDEX_FILE):
        try:
            with open(INDEX_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load project index: {e}")

    # Default index
    index = {
        "last_updated": datetime.datetime.now().isoformat(),
        "projects": {}
    }

    # Save default index
    save_project_index(index)
    return index


def save_project_index(index: Dict[str, Any]) -> None:
    """Save project index to file."""
    try:
        os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
        # Update timestamp
        index["last_updated"] = datetime.datetime.now().isoformat()
        with open(INDEX_FILE, 'w') as f:
            json.dump(index, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save project index: {e}")


def convert_to_snake_case(name: str) -> str:
    """Convert a string to snake_case."""
    # Replace spaces and hyphens with underscores
    s = name.replace(' ', '_').replace('-', '_')
    # Remove any non-alphanumeric characters (except underscores)
    s = ''.join(c for c in s if c.isalnum() or c == '_')
    # Convert to lowercase
    return s.lower()


def create_project(
    project_name: str,
    template: str = "default",
    description: str = "",
    path: str = None,
    open_vs_code: bool = True
) -> str:
    """
    Create a new project from a template.

    Args:
        project_name: Name of the project
        template: Template to use
        description: Project description
        path: Path to create the project (if None, use config)
        open_vs_code: Whether to open the project in VS Code when done

    Returns:
        Path to the created project
    """
    # Load configuration
    config = load_config()

    # Get the template
    if template not in TEMPLATE_STRUCTURE:
        logger.error(f"Template '{template}' not found")
        available_templates = ", ".join(TEMPLATE_STRUCTURE.keys())
        logger.info(f"Available templates: {available_templates}")
        return ""

    # Set up project path
    if path is None:
        path = config["projects_dir"]

    # Make sure path exists
    os.makedirs(path, exist_ok=True)

    # Create project directory
    project_path = os.path.join(path, project_name)
    if os.path.exists(project_path):
        logger.error(f"Project directory already exists: {project_path}")
        return ""

    # Create project directory
    os.makedirs(project_path, exist_ok=True)
    logger.info(f"Created project directory: {project_path}")

    # Prepare template data for substitution
    project_name_snake = convert_to_snake_case(project_name)
    date = datetime.datetime.now().strftime("%B %d, %Y")
    year = datetime.datetime.now().year

    template_data = {
        "project_name": project_name,
        "project_name_snake": project_name_snake,
        "description": description,
        "date": date,
        "year": year,
        "author": config["author"],
        "email": config["git_email"],
        "git_username": config["git_username"]
    }

    # Get the complete template structure
    template_structure = get_complete_template(template)

    # Create directories
    for directory in template_structure["directories"]:
        # Replace template variables in directory name
        dir_path = os.path.join(
            project_path,
            directory.format(**template_data)
        )
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

    # Create files
    for file_path, content in template_structure["files"].items():
        # Replace template variables in file path
        full_path = os.path.join(
            project_path,
            file_path.format(**template_data)
        )

        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        # Replace template variables in content
        file_content = content.format(**template_data)

        # Write file
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(file_content)

        logger.info(f"Created file: {full_path}")

    # Initialize git repository
    if template_structure.get("git_init", False) and config.get("git_enabled", True):
        try:
            subprocess.run(
                ["git", "init"],
                cwd=project_path,
                check=True,
                capture_output=True
            )

            # Set user info
            subprocess.run(
                ["git", "config", "user.name", config["author"]],
                cwd=project_path,
                check=True,
                capture_output=True
            )

            subprocess.run(
                ["git", "config", "user.email", config["git_email"]],
                cwd=project_path,
                check=True,
                capture_output=True
            )

            # Add all files
            subprocess.run(
                ["git", "add", "."],
                cwd=project_path,
                check=True,
                capture_output=True
            )

            # Initial commit
            subprocess.run(
                ["git", "commit", "-m", "Initial commit"],
                cwd=project_path,
                check=True,
                capture_output=True
            )

            logger.info("Initialized git repository")
        except Exception as e:
            logger.warning(f"Failed to initialize git repository: {e}")

    # Create VS Code settings
    if config.get("vs_code_enabled", True):
        vs_code_dir = os.path.join(project_path, ".vscode")
        os.makedirs(vs_code_dir, exist_ok=True)

        # Create settings.json
        settings = template_structure.get("vs_code", {}).get("settings", {})
        if settings:
            with open(os.path.join(vs_code_dir, "settings.json"), 'w') as f:
                json.dump(settings, f, indent=2)

        # Create extensions.json
        extensions = template_structure.get("vs_code", {}).get("extensions", [])
        if extensions:
            with open(os.path.join(vs_code_dir, "extensions.json"), 'w') as f:
                json.dump({"recommendations": extensions}, f, indent=2)

        logger.info("Created VS Code settings")

    # Update project index
    index = load_project_index()
    index["projects"][project_name] = {
        "path": project_path,
        "template": template,
        "description": description,
        "created": datetime.datetime.now().isoformat(),
        "last_modified": datetime.datetime.now().isoformat()
    }
    save_project_index(index)

    # Open in VS Code if requested
    if open_vs_code and config.get("auto_open_vs_code", True):
        try:
            subprocess.Popen(["code", project_path])
            logger.info("Opened project in VS Code")
        except Exception as e:
            logger.warning(f"Failed to open project in VS Code: {e}")

    logger.info(f"Project {project_name} created successfully at {project_path}")
    return project_path


def get_complete_template(template_name: str) -> Dict[str, Any]:
    """
    Get the complete template structure, resolving inheritance.

    Args:
        template_name: Name of the template

    Returns:
        Complete template structure
    """
    if template_name not in TEMPLATE_STRUCTURE:
        raise ValueError(f"Template '{template_name}' not found")

    template = TEMPLATE_STRUCTURE[template_name]

    # Check if the template extends another template
    if "extends" in template:
        parent_template_name = template["extends"]
        parent_template = get_complete_template(parent_template_name)

        # Merge directories
        merged_dirs = parent_template.get("directories", []).copy()
        merged_dirs.extend(template.get("directories", []))

        # Merge files
        merged_files = parent_template.get("files", {}).copy()
        merged_files.update(template.get("files", {}))

        # Merge dependencies
        merged_deps = parent_template.get("dependencies", []).copy()
        merged_deps.extend(template.get("dependencies", []))

        # Merge VS Code settings
        merged_vs_code = parent_template.get("vs_code", {}).copy()
        if "vs_code" in template:
            if "extensions" in template["vs_code"]:
                merged_extensions = merged_vs_code.get("extensions", []).copy()
                merged_extensions.extend(template["vs_code"].get("extensions", []))
                merged_vs_code["extensions"] = merged_extensions

            if "settings" in template["vs_code"]:
                merged_settings = merged_vs_code.get("settings", {}).copy()
                merged_settings.update(template["vs_code"].get("settings", {}))
                merged_vs_code["settings"] = merged_settings

        # Create merged template
        merged_template = {
            "directories": merged_dirs,
            "files": merged_files,
            "dependencies": merged_deps,
            "vs_code": merged_vs_code,
            "git_init": template.get("git_init", parent_template.get("git_init", False))
        }

        return merged_template

    # If the template doesn't extend another template, return it as is
    return template


def generate_project_dashboard() -> str:
    """
    Generate a dashboard of all projects.

    Returns:
        Path to the generated dashboard
    """
    # Load project index
    index = load_project_index()
    config = load_config()

    # Dashboard path
    dashboard_path = os.path.join(config["projects_dir"], "PROJECT_DASHBOARD.md")

    # Generate dashboard content
    content = f"# Project Dashboard\n\n"
    content += f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    if not index["projects"]:
        content += "No projects found.\n"
    else:
        content += "| Project | Template | Description | Created | Last Modified |\n"
        content += "| ------- | -------- | ----------- | ------- | ------------- |\n"

        for name, project in index["projects"].items():
            created = datetime.datetime.fromisoformat(project["created"]).strftime("%Y-%m-%d")
            modified = datetime.datetime.fromisoformat(project["last_modified"]).strftime("%Y-%m-%d")

            content += f"| [{name}]({project['path']}) | {project['template']} | {project['description']} | {created} | {modified} |\n"

    # Write dashboard
    with open(dashboard_path, 'w') as f:
        f.write(content)

    logger.info(f"Generated project dashboard at {dashboard_path}")
    return dashboard_path


def create_project_gui() -> None:
    """Launch a simple GUI for project creation."""
    try:
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
    except ImportError:
        logger.error("Tkinter is not available. Please install Python with Tkinter support.")
        return

    # Load configuration
    config = load_config()

    # Create main window
    root = tk.Tk()
    root.title("Project Generator")
    root.geometry("600x500")
    root.resizable(True, True)

    # Frame for the form
    frame = ttk.Frame(root, padding="20")
    frame.pack(fill=tk.BOTH, expand=True)

    # Project name
    ttk.Label(frame, text="Project Name:").grid(column=0, row=0, sticky=tk.W, pady=5)
    project_name = ttk.Entry(frame, width=40)
    project_name.grid(column=1, row=0, sticky=(tk.W, tk.E), pady=5)
    project_name.focus()

    # Template
    ttk.Label(frame, text="Template:").grid(column=0, row=1, sticky=tk.W, pady=5)
    template_var = tk.StringVar()
    template_combo = ttk.Combobox(frame, textvariable=template_var, width=38)
    template_combo['values'] = list(TEMPLATE_STRUCTURE.keys())
    template_combo.current(0)
    template_combo.grid(column=1, row=1, sticky=(tk.W, tk.E), pady=5)

    # Description
    ttk.Label(frame, text="Description:").grid(column=0, row=2, sticky=tk.W, pady=5)
    description = tk.Text(frame, width=40, height=4)
    description.grid(column=1, row=2, sticky=(tk.W, tk.E), pady=5)

    # Path
    ttk.Label(frame, text="Project Path:").grid(column=0, row=3, sticky=tk.W, pady=5)
    path_var = tk.StringVar(value=config["projects_dir"])
    path_entry = ttk.Entry(frame, textvariable=path_var, width=30)
    path_entry.grid(column=1, row=3, sticky=(tk.W, tk.E), pady=5)

    def browse_path():
        directory = filedialog.askdirectory(initialdir=config["projects_dir"])
        if directory:
            path_var.set(directory)

    browse_button = ttk.Button(frame, text="Browse...", command=browse_path)
    browse_button.grid(column=2, row=3, sticky=tk.W, pady=5)

    # Open in VS Code
    open_vs_code_var = tk.BooleanVar(value=config["auto_open_vs_code"])
    open_vs_code_check = ttk.Checkbutton(
        frame,
        text="Open in VS Code when done",
        variable=open_vs_code_var
    )
    open_vs_code_check.grid(column=1, row=4, sticky=tk.W, pady=5)

    # Status
    status_var = tk.StringVar()
    status_label = ttk.Label(frame, textvariable=status_var, wraplength=500)
    status_label.grid(column=0, row=6, columnspan=3, sticky=(tk.W, tk.E), pady=10)

    # Template info
    template_info_var = tk.StringVar()
    template_info = ttk.Label(frame, textvariable=template_info_var, wraplength=500)
    template_info.grid(column=0, row=7, columnspan=3, sticky=(tk.W, tk.E), pady=5)

    def update_template_info(*args):
        """Update the template info text when the template selection changes."""
        template_name = template_var.get()
        if template_name in TEMPLATE_STRUCTURE:
            template = TEMPLATE_STRUCTURE[template_name]
            extends = f" (extends {template['extends']})" if "extends" in template else ""
            dependencies = ", ".join(template.get("dependencies", []))
            extensions = ", ".join(template.get("vs_code", {}).get("extensions", []))

            info = f"Template: {template_name}{extends}\n"
            if dependencies:
                info += f"Dependencies: {dependencies}\n"
            if extensions:
                info += f"VS Code Extensions: {extensions}"

            template_info_var.set(info)

    # Update template info when selection changes
    template_var.trace_add("write", update_template_info)
    update_template_info()  # Initial update

    def create_project_action():
        """Create the project when the Create button is clicked."""
        name = project_name.get().strip()
        template = template_var.get()
        desc = description.get("1.0", tk.END).strip()
        path = path_var.get()
        open_vscode = open_vs_code_var.get()

        if not name:
            messagebox.showerror("Error", "Project name is required")
            return

        status_var.set("Creating project...")
        root.update()

        try:
            project_path = create_project(
                project_name=name,
                template=template,
                description=desc,
                path=path,
                open_vs_code=open_vscode
            )

            if project_path:
                status_var.set(f"Project created successfully at {project_path}")
                messagebox.showinfo("Success", f"Project created successfully at {project_path}")

                # Clear form for next project
                project_name.delete(0, tk.END)
                description.delete("1.0", tk.END)
            else:
                status_var.set("Failed to create project")
                messagebox.showerror("Error", "Failed to create project")
        except Exception as e:
            status_var.set(f"Error: {e}")
            messagebox.showerror("Error", f"An error occurred: {e}")

    # Buttons
    button_frame = ttk.Frame(frame)
    button_frame.grid(column=0, row=5, columnspan=3, pady=10)

    create_button = ttk.Button(button_frame, text="Create Project", command=create_project_action)
    create_button.pack(side=tk.LEFT, padx=5)

    def generate_dashboard_action():
        """Generate the project dashboard when the button is clicked."""
        try:
            dashboard_path = generate_project_dashboard()
            status_var.set(f"Dashboard generated at {dashboard_path}")
            messagebox.showinfo("Success", f"Dashboard generated at {dashboard_path}")
        except Exception as e:
            status_var.set(f"Error generating dashboard: {e}")
            messagebox.showerror("Error", f"Error generating dashboard: {e}")

    dashboard_button = ttk.Button(
        button_frame,
        text="Generate Dashboard",
        command=generate_dashboard_action
    )
    dashboard_button.pack(side=tk.LEFT, padx=5)

    cancel_button = ttk.Button(button_frame, text="Close", command=root.destroy)
    cancel_button.pack(side=tk.LEFT, padx=5)

    # Run the application
    root.mainloop()


def main() -> int:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Universal Project Generator and Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python project_generator.py create --name "My Project" --template python
  python project_generator.py create --name "Research Project" --template research --desc "A scientific research project"
  python project_generator.py dashboard
  python project_generator.py gui
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new project")
    create_parser.add_argument("--name", "-n", required=True, help="Project name")
    create_parser.add_argument(
        "--template", "-t",
        default="default",
        choices=TEMPLATE_STRUCTURE.keys(),
        help="Project template"
    )
    create_parser.add_argument("--desc", "-d", default="", help="Project description")
    create_parser.add_argument("--path", "-p", help="Project parent directory")
    create_parser.add_argument(
        "--no-vscode",
        action="store_true",
        help="Don't open in VS Code when done"
    )

    # Dashboard command
    subparsers.add_parser("dashboard", help="Generate project dashboard")

    # Config command
    config_parser = subparsers.add_parser("config", help="Configure project generator")
    config_parser.add_argument("--projects-dir", help="Default projects directory")
    config_parser.add_argument("--templates-dir", help="Templates directory")
    config_parser.add_argument("--author", help="Default author name")
    config_parser.add_argument("--git-username", help="Git username")
    config_parser.add_argument("--git-email", help="Git email")
    config_parser.add_argument(
        "--disable-vscode",
        action="store_true",
        help="Disable VS Code integration"
    )
    config_parser.add_argument(
        "--disable-git",
        action="store_true",
        help="Disable Git initialization"
    )
    config_parser.add_argument(
        "--disable-auto-open",
        action="store_true",
        help="Disable automatically opening projects in VS Code"
    )

    # GUI command
    subparsers.add_parser("gui", help="Launch GUI for project creation")

    # List command
    subparsers.add_parser("list", help="List all projects")

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == "create":
            project_path = create_project(
                project_name=args.name,
                template=args.template,
                description=args.desc,
                path=args.path,
                open_vs_code=not args.no_vscode
            )

            if not project_path:
                return 1

        elif args.command == "dashboard":
            generate_project_dashboard()

        elif args.command == "config":
            config = load_config()

            # Update config with provided values
            if args.projects_dir:
                config["projects_dir"] = os.path.abspath(args.projects_dir)

            if args.templates_dir:
                config["templates_dir"] = os.path.abspath(args.templates_dir)

            if args.author:
                config["author"] = args.author

            if args.git_username:
                config["git_username"] = args.git_username

            if args.git_email:
                config["git_email"] = args.git_email

            if args.disable_vscode:
                config["vs_code_enabled"] = False

            if args.disable_git:
                config["git_enabled"] = False

            if args.disable_auto_open:
                config["auto_open_vs_code"] = False

            # Save updated config
            save_config(config)
            logger.info("Configuration updated")

        elif args.command == "gui":
            create_project_gui()

        elif args.command == "list":
            index = load_project_index()

            if not index["projects"]:
                print("No projects found")
            else:
                print("\nProjects:")
                print("=========")

                for name, project in index["projects"].items():
                    created = datetime.datetime.fromisoformat(project["created"]).strftime("%Y-%m-%d")
                    print(f"{name} ({project['template']}) - {created}")
                    print(f"  Path: {project['path']}")
                    if project['description']:
                        print(f"  Description: {project['description']}")
                    print()

        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
