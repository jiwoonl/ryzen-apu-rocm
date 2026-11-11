#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prepare a split ROCm SDK development overlay without changing runtime wheels."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import sysconfig

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--directory', type=Path, default=Path('build/ck-env'))
p.add_argument('--index-url',default='https://stable.repo.amd.com/rocm/whl-next/')
a=p.parse_args();folder=a.directory.resolve()
subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(folder)],check=True)
py=folder/'bin/python'
subprocess.run([str(py),'-m','pip','install','--index-url',a.index_url,'rocm-sdk-devel==10.1.0'],check=True)
site=Path(subprocess.check_output([str(py),'-c','import sysconfig;print(sysconfig.get_paths()["purelib"])'],text=True).strip())
runtime_site=Path(sysconfig.get_paths()['purelib'])
for name in ('_rocm_sdk_core','_rocm_sdk_libraries'):
    target=runtime_site/name;link=site/name
    if not target.is_dir():raise SystemExit(f'Missing runtime SDK directory: {name}')
    if not link.exists():link.symlink_to(target, target_is_directory=True)
print('ROCM_DEVEL='+str(site/'_rocm_sdk_devel'))
