#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build PyXRT for the active Python using an existing matching XRT install."""
import argparse
from pathlib import Path
import os
import subprocess
import sysconfig
import torch

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True, help='XRT source checkout')
p.add_argument('--xrt-prefix', type=Path, default=Path.home()/'.local/opt/xrt')
p.add_argument('--output-dir', type=Path, default=Path('build/pyxrt'))
p.add_argument('--install', action='store_true', help='write into active Python site-packages')
a = p.parse_args()
prefix = a.xrt_prefix.expanduser().resolve()
source = a.source.resolve()/'src/python/pybind11/src/pyxrt.cpp'
if not source.is_file() or not (prefix/'include/xrt/xrt_device.h').is_file():
    p.error('XRT source and installed headers are required')
folder = Path(sysconfig.get_paths()['platlib']) if a.install else a.output_dir
folder.mkdir(parents=True, exist_ok=True)
out = folder/('pyxrt'+sysconfig.get_config_var('EXT_SUFFIX'))
cmd = [os.environ.get('CXX', 'g++'), '-O2', '-std=c++17', '-shared', '-fPIC',
       '-I'+sysconfig.get_paths()['include'], '-I'+str(Path(torch.__file__).parent/'include'),
       '-I'+str(prefix/'include'), str(source), '-L'+str(prefix/'lib64'),
       '-Wl,-rpath,'+str(prefix/'lib64'), '-lxrt_coreutil', '-luuid', '-pthread', '-o', str(out)]
subprocess.run(cmd, check=True)
print(out)
