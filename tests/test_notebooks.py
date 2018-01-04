"""
Tests for plots must be visually inspected. This script loads and executes all
cells in ipython notebooks to check code coverage with pytest.
"""

import io
import os
import re

from IPython import get_ipython
from IPython.core.interactiveshell import InteractiveShell
import nbformat

PATH = 'tests/'

notebook_filenames = []
for filename in os.listdir(PATH):
    if re.match('test.*\.ipynb$', filename):
        notebook_filenames.append(filename)

for filename in notebook_filenames:
    full_path = PATH + filename
    with io.open(full_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, 4)

    shell = InteractiveShell.instance()

    # Inline matplotlib cannot be rendered outside of jupyter notebooks.
    shell.enable_gui = lambda x: False
    for cell in nb.cells:
        if cell.cell_type == 'code':
            code = shell.input_transformer_manager.transform_cell(cell.source)
            exec(code)
