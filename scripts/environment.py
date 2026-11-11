#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Print a hardware/software snapshot without host names or user paths."""
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import subprocess
import torch

names=('torch','torchvision','torchaudio','triton','rocm','rocm-sdk-core',
       'rocm-sdk-libraries','rocm-sdk-device-gfx1103','amd-torch-device-gfx1103',
       'amd-torch-device-gfx110x','amd-torchvision-device-gfx1103','mlir-aie','llvm-aie','numpy','ml-dtypes')
versions={}
for name in names:
    try:versions[name]=metadata.version(name)
    except metadata.PackageNotFoundError:pass
cpu=next(line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name'))
mem=int(Path('/proc/meminfo').read_text().splitlines()[0].split()[1])
report=dict(machine=Path('/sys/class/dmi/id/product_version').read_text().strip(),
    cpu=cpu,logical_cpus=len(__import__('os').sched_getaffinity(0)),memory_kib=mem,
    os=platform.freedesktop_os_release().get('PRETTY_NAME'),kernel=platform.release(),
    glibc=platform.libc_ver()[1],python=platform.python_version(),packages=versions)
if torch.cuda.is_available():
    props=torch.cuda.get_device_properties(0)
    report['gpu']=dict(name=props.name,architecture=props.gcnArchName,hip=torch.version.hip)
report['xrt_commit']='fdb373b1a43cd44201c799b80e2dd38df68e86b1'
report['xdna_driver_source_commit']='5599041841d8522e8f206b6fc55e0230f8cac915'
report['miopen_source_commit']='39d8d60365d2272a5366f76ccb3bfba24b219eaf'
report['mlir_aie_source_commit']='ed23bba7381b6f0680775fd3b6110d9268d27f43'
# Source revisions above describe this repository's reference setup, not a probe.
report['source_revision_note']='Pinned reference sources; scripts do not infer local source revisions.'
print(json.dumps(report,indent=2))
