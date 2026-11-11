#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Keep MLIR's LLVM symbols local when torch was imported first (Linux)."""
import importlib.metadata
from pathlib import Path
import shutil

path = Path(importlib.metadata.distribution('mlir-aie').locate_file('mlir_aie/python/aie/_mlir_libs/__init__.py'))
source = path.read_text()
if 'RTLD_DEEPBIND' in source:
    print('MLIR loader already patched')
else:
    needle = '\n_site_initialize()\n'
    if source.count(needle) != 1:
        raise SystemExit(f'Unexpected MLIR loader layout: {path}')
    replacement = '''
import os as _aie_os
import sys as _aie_sys
_aie_loader_flags = _aie_sys.getdlopenflags()
try:
    if hasattr(_aie_os, "RTLD_DEEPBIND"):
        _aie_sys.setdlopenflags(_aie_loader_flags | _aie_os.RTLD_DEEPBIND)
    _site_initialize()
finally:
    _aie_sys.setdlopenflags(_aie_loader_flags)
    del _aie_loader_flags, _aie_os, _aie_sys
'''
    backup = path.with_suffix('.py.pre-deepbind')
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(source.replace(needle, '\n' + replacement))
    print('Patched MLIR loader; original saved next to it')
