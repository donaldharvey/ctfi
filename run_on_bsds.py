import sh
import pathlib

cmd = sh.Command('./ctf.py')

for p in pathlib.Path('/me/w/proj/slic/SLIC-Superpixels/BSDS500').glob("*.jpg"):
    print p.stem
    out = cmd(p)
    with open(p.stem + ".json", 'w') as f:
        f.write(out)
