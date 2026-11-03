"""
Run once per conda environment after cloning:

    conda activate cdi
    python setup_path.py

Writes a .pth file into the active environment's site-packages so that
`import runup` works from anywhere. No package installation required.
Edits to source files take effect immediately without reinstalling.
"""
import site
import pathlib

repo_root = pathlib.Path(__file__).resolve().parent
pth_file = pathlib.Path(site.getsitepackages()[0]) / "runup_repo.pth"
pth_file.write_text(str(repo_root))
print(f"Registered: {repo_root}")
print(f"Written to: {pth_file}")
print("Done — 'import runup' will now work in this environment.")
