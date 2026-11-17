import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen
import zipfile

spec=json.loads(Path('/tmp/vendors.lock.json').read_text())['gigaam']
with urlopen(spec['url'],timeout=60) as response:
    data=response.read()
if hashlib.sha256(data).hexdigest()!=spec['sha256']:
    raise ValueError('GigaAM source archive checksum mismatch')
root=Path('/opt/gigaam')
with zipfile.ZipFile(io.BytesIO(data)) as archive:
    for name in archive.namelist():
        parts=Path(name).parts[1:]
        if not parts or '..' in parts or name.endswith('/'):
            continue
        if parts[0]!='gigaam' and parts!=('LICENSE',):
            continue
        target=root.joinpath(*parts)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(archive.read(name))
