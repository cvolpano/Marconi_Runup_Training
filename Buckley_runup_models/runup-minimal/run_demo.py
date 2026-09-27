"""
Molokai demo — runs the full pipeline on the bundled test files.

Steps:
    1. split_for_segmentation  — prepare images for ML model
    2. run_segmentation_onnx   — generate softmax scores
    3. timestack2netcdf        — convert to NetCDF
    4. check_nc_images         — QA/QC figures

All steps run in the runup conda environment.

Expected input files in runup_demo_files/input/:
    <epoch>.<...>.molokai.c2.runup03.mat        timestack
    topoSurvey_lidar_elevationCapped.mat         topo survey (single undated file)
    molokai.c2.r.mat                             r-file (XYZ pixel coords)
    UV.txt  OR  molokai.c2.pix                   UV pixel coords

YAML calibration files must be in the repo's calibration/ directory:
    MolokaiHI_c2_<YYYYMMDD>_EO.yaml
    MolokaiHI_c2_<YYYYMMDD>_IO.yaml
    MolokaiHI_c2_<YYYYMMDD>_metadata.yaml
    MolokaiHI_localOrigin.yaml

Usage:
    conda activate runup
    python run_demo.py
"""

import shutil
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
SITE_NAME  = 'molokai_demo'
DEMO_ROOT  = Path(__file__).resolve().parent / 'runup_demo_files'
CONFIG_DIR = str(Path(__file__).resolve().parent / 'config')

# ── Directory layout ──────────────────────────────────────────────────────────
INPUT_DIR  = DEMO_ROOT / 'input'
OUTPUT_DIR = DEMO_ROOT / 'output'

DIRS = {
    'seg':     OUTPUT_DIR / 'seg',
    'netcdf':  OUTPUT_DIR / 'netcdf',
    'figures': OUTPUT_DIR / 'figures',
    'logs':    OUTPUT_DIR / 'logs',
}


def setup_dirs():
    """Clear and recreate the output directory tree."""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)
    print('Output directories ready:')
    for name, path in DIRS.items():
        print(f'  {name:8s}  {path}')
    print()


def diagnose():
    """Print diagnostic info to confirm files are found before running."""
    from runup.pixel2runup import getSite
    site = getSite(SITE_NAME, CONFIG_DIR)
    site.gatherTimestacks()

    print(f'Looking for timestacks in : {site.stackDir}')
    print(f'Matching extension        : {site.stackExtension}')
    print(f'Matching name fragment    : {site.nName}')
    print(f'Files in stackDir         : {[f.name for f in Path(site.stackDir).iterdir()]}')
    print()
    print(f'Timestacks found: {len(site.timestackList)}')
    for f in site.timestackList:
        print(f'  {f}')
    print()

    for f in site.timestackList:
        filename = Path(f).name
        parts    = filename.split('.')
        print(f'Filename : {filename}')
        print(f'parts[7] : {parts[7]}  (camNum source)')
        print(f'camNum   : {parts[7][-1]}')


if __name__ == '__main__':
    print('Molokai demo pipeline')
    print(f'Site:    {SITE_NAME}')
    print(f'Config:  {CONFIG_DIR}/{SITE_NAME}_config.json')
    print(f'Root:    {DEMO_ROOT}')
    print()

    setup_dirs()
    diagnose()

    print('=' * 60)
    print('Step 1: split_for_segmentation')
    print('=' * 60)
    from runup.split_for_segmentation import split_for_segmentation
    split_for_segmentation(SITE_NAME)
    print()

    print('=' * 60)
    print('Step 2: run_segmentation_onnx')
    print('=' * 60)
    from runup.run_segmentation_onnx import run_segmentation_onnx
    run_segmentation_onnx(SITE_NAME)
    print()

    print('=' * 60)
    print('Step 3: timestack2netcdf')
    print('=' * 60)
    from runup.timestack2netcdf import timestack2netcdf
    timestack2netcdf(SITE_NAME)
    print()

    print('=' * 60)
    print('Step 4: check_nc_images')
    print('=' * 60)
    from runup.check_nc_images import check_nc_images
    check_nc_images(SITE_NAME)
    print()

    print('=' * 60)
    print('Demo complete.')
    print(f'  NetCDF files : {DIRS["netcdf"]}')
    print(f'  QA/QC figures: {DIRS["figures"]}')
    print(f'  Logs         : {DIRS["logs"]}')
