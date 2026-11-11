#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Download pinned Peano and fix its inconsistent internal wheel tags."""
import base64
import csv
import hashlib
import io
from pathlib import Path
import urllib.request
import zipfile

name = 'llvm_aie-21.0.0.2026062301+cb664e8c-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl'
url = 'https://github.com/Xilinx/llvm-aie/releases/download/nightly/' + name.replace('+', '%2B')
folder = Path('wheels')
folder.mkdir(exist_ok=True)
original = folder / (name + '.download')
target = folder / name
if not original.exists():
    urllib.request.urlretrieve(url, original)
with zipfile.ZipFile(original) as archive:
    wheel = next(n for n in archive.namelist() if n.endswith('.dist-info/WHEEL'))
    record = next(n for n in archive.namelist() if n.endswith('.dist-info/RECORD'))
    text = archive.read(wheel).decode()
    lines = [line for line in text.splitlines() if not line.startswith('Tag:')]
    lines += ['Tag: py3-none-manylinux_2_27_x86_64', 'Tag: py3-none-manylinux_2_28_x86_64']
    data = ('\n'.join(lines) + '\n').encode()
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('=')
    rows = list(csv.reader(io.StringIO(archive.read(record).decode())))
    for row in rows:
        if row[0] == wheel:
            row[1:] = ['sha256=' + digest, str(len(data))]
    output = io.StringIO(newline='')
    csv.writer(output).writerows(rows)
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as out:
        for entry in archive.infolist():
            out.writestr(entry, data if entry.filename == wheel else output.getvalue().encode() if entry.filename == record else archive.read(entry.filename))
print(target)
