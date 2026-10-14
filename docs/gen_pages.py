"""Generated pages for the MkDocs site (mkdocs-gen-files): the lab log from exp/NOTES.md, and the exp/out figures
the results pages embed, so that the site builds from docs/ alone. Links into these are rewritten by docs/hooks.py."""
import pathlib

import mkdocs_gen_files

ROOT = pathlib.Path(__file__).resolve().parents[1]

with mkdocs_gen_files.open("LabLog.md", "w") as f:
    f.write((ROOT / "exp" / "NOTES.md").read_text())

for png in sorted((ROOT / "exp" / "out").glob("*.png")):
    with mkdocs_gen_files.open(f"exp_out/{png.name}", "wb") as f:
        f.write(png.read_bytes())
