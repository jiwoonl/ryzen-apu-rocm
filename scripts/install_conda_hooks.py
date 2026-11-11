#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Install environment-scoped CK/XRT hooks in the active conda environment."""
import os
from pathlib import Path
import shutil

prefix = os.environ.get('CONDA_PREFIX')
if not prefix:
    raise SystemExit('Activate the intended conda environment first')
source = Path(__file__).resolve().parent/'conda'
for kind in ('activate.d', 'deactivate.d'):
    folder = Path(prefix)/'etc/conda'/kind
    folder.mkdir(parents=True, exist_ok=True)
    for file in (source/kind).glob('*.sh'):
        target = folder/file.name
        if target.exists() and target.read_bytes() != file.read_bytes():
            backup = target.with_suffix('.sh.before-apu-rocm')
            if not backup.exists():
                shutil.copy2(target, backup)
        shutil.copy2(file, target)
print('Hooks installed. Deactivate and reactivate this environment.')
