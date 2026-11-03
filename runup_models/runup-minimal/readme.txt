# runup

Python tools for processing CoastCam timestack images into NetCDF files containing runup and total water level data.

---

## Repository layout

```
runup/
├── runup/                          importable Python package
│   ├── pixel2runup.py                  core functions and Site class
│   ├── download_timestacks.py          step 1
│   ├── isolate_timestacks.py           step 2
│   ├── split_for_segmentation.py       step 3
│   ├── run_segmentation_onnx.py        step 4
│   ├── timestack2netcdf.py             step 5
│   ├── check_nc_images.py              QA/QC figures
│   └── extra_functions.py              archived/reference functions
├── config/                         JSON site configuration files
├── calibration/                    YAML camera calibration files (EO, IO, metadata, localOrigin)
├── envs/                           conda environment files
├── models/                         ONNX model weights
├── runup_demo_files/               bundled demo data (Molokai)
│   ├── input/                          timestacks, topo, UV, r-file (tracked by git)
│   └── output/                         seg, netcdf, figures, logs (gitignored)
├── logs/                           pipeline log output (gitignored)
├── run_demo.py                     end-to-end demo script
└── setup_path.py                   one-time path registration script
```

---

## Dependencies

All dependencies are contained within this repo and the `runup` conda environment. There are no sibling-repo requirements.

---

## Environment setup

All pipeline steps use a single conda environment.

```bash
conda env create -f envs/runup.yml
conda activate runup
python setup_path.py
```

`setup_path.py` writes a `.pth` file into the active environment's site-packages so that `import runup` works from anywhere without installing a package. Edits to source files take effect immediately.

---

## Running the demo

The repo includes a self-contained Molokai demo with all required input files bundled under `runup_demo_files/input/`. The demo clears and recreates `runup_demo_files/output/` on each run.

```bash
conda activate runup
python run_demo.py
```

Output is written to:

| Path | Contents |
|---|---|
| `runup_demo_files/output/netcdf/` | NetCDF files |
| `runup_demo_files/output/figures/` | QA/QC figures |
| `runup_demo_files/output/seg/` | Softmax `.npy` files |
| `runup_demo_files/output/logs/` | Log files |

---

## Full pipeline workflow

Each step is a Python module that can be run from the command line or imported directly. Run them in order, passing the site name as an argument:

```bash
conda activate runup

python -m runup.download_timestacks MadBeach2022
python -m runup.isolate_timestacks Dorado
python -m runup.split_for_segmentation MadBeach2022
python -m runup.run_segmentation_onnx MadBeach2022
python -m runup.timestack2netcdf Molokai
python -m runup.check_nc_images Molokai
```

Steps can also be called directly from Python:

```python
from runup.download_timestacks import download_timestacks
from runup.timestack2netcdf import timestack2netcdf

download_timestacks('MadBeach2022')
timestack2netcdf('Molokai')
```

Looping over multiple sites:

```python
from runup.timestack2netcdf import timestack2netcdf

for site in ['Dorado', 'Molokai', 'TresPalmasPR']:
    timestack2netcdf(site)
```

### Available sites

| Site name | Description |
|---|---|
| `Dorado` | Dorado Beach, Puerto Rico |
| `IslaVerde` | Isla Verde, Puerto Rico |
| `IslaVerdePR` | Isla Verde PR (legacy calibration) |
| `MadBeach2017` | Madeira Beach, Florida (2017) |
| `MadBeach2022` | Madeira Beach, Florida (2022) |
| `Molokai` | Molokai, Hawaii |
| `SandKey` | Sand Key, Florida |
| `TresPalmasPR` | Tres Palmas, Puerto Rico |

---

## Site configuration files

Each site has a JSON config file in `config/` with three sections: `directories`, `metadata`, and `flags`.

Directory paths are resolved relative to the repo root, so both relative paths (`runup_demo_files/input`) and absolute network paths (`//gs/stpetersburgfl-g/...`) are supported. Forward and back slashes are normalised to the OS separator at load time.

### `directories`

| Key | Description |
|---|---|
| `netcdfDir` | Output directory for NetCDF files |
| `figureDir` | Output directory for QA/QC figures |
| `topoDir` | Directory containing topo survey files |
| `uvTopDir` | Top-level directory for UV pixel coordinate files |
| `xyzTopDir` | Top-level directory for XYZ / r-file coordinate data |
| `rawDirTop` | Top-level directory for raw Argus archive (year/cx/day structure). If non-empty, used as the timestack source instead of `stackDir` |
| `stackDir` | Directory for processed timestack files. Used when `rawDirTop` is `""` |
| `segDir` | Directory for softmax segmentation `.npy`/`.npz` files |
| `modelDir` | Directory containing SegmentationGym model weights |
| `s3TopDir` | S3 URI for downloading raw timestacks in step 1 (e.g. `s3://cmgp-coastcam/cameras/molokai`) |

### `metadata`

| Key | Description |
|---|---|
| `saveName` | Short name used to match YAML calibration files (e.g. `MolokaiHI`, `madbeach`) |
| `siteLocation` | Human-readable site location for NetCDF attributes |
| `dataOrigin` | Data origin center for NetCDF attributes |
| `contributors` | Contributor names for NetCDF attributes |
| `references` | Citation string for NetCDF attributes |
| `camNum` | Default camera number. Only needed for `cx` sites where the camera number cannot be parsed from the filename. For all `c#` sites this is derived automatically from `exampleStackName` |
| `camMake` | Camera manufacturer |
| `camModel` | Camera model |
| `camLens` | Camera lens model |
| `timezone` | Local timezone string (e.g. `HST`, `EST`) |
| `utmZone` | UTM zone string (e.g. `04Q`, `17R`) |
| `verticalDatum` | Short vertical datum name written to NetCDF (e.g. `NAVD88`, `LMSL`) |
| `verticalDatum_description` | Full vertical datum description for NetCDF attributes |

### `flags`

| Key | Description |
|---|---|
| `exampleStackName` | A real example post-split timestack filename. Used to automatically derive `stackExtension`, `nName`, `camNum`, and `rNumber` |
| `exampleTopoName` | Topo survey filename. Use `<date>` as a token where `YYYYMMDD` appears (e.g. `"topoSurveyInterpMaskedAllTopo<date>_lidar_elevationCapped.mat"`). Omit `<date>` for a single undated file. Supports `.mat` and `.xyz` formats |
| `exampleRFileName` | Example r-file filename. Supports the `<date>` token. If omitted, the code looks for `<nName>.c<N>.r.mat` in the closest `pix_*` subdirectory |
| `startDate` | Start of processing date range (`YYYY-MM-DD`, UTC) |
| `endDate` | End of processing date range (`YYYY-MM-DD`, UTC) |
| `writeOverNC` | `1` to overwrite existing NetCDF files, `0` to skip them |
| `tiff_sample_frequency_Hz` | Sample frequency of `.tiff` timestacks in Hz (required for `.tiff` sites) |
| `skipCams` | List of camera numbers to skip (e.g. `[2]`) |
| `stackNumMap` | Dict mapping legacy stack numbers to sequential ones (e.g. `{"90": 1, "150": 2, "250": 3}`) |
| `xyzPreTrimmed` | `true` if XYZ arrays are already trimmed to each stack's extent. Default `false` |
| `modelWeights` | Filename of the ONNX model weights file within `modelDir` |

---

## Adding a new site

1. Add `config/<SiteName>_config.json` following the structure above.
2. Add YAML calibration files to `calibration/` following the naming convention `<saveName>_c<camNum>_<YYYYMMDD>_EO.yaml`, `_IO.yaml`, `_metadata.yaml`, and `<saveName>_localOrigin.yaml`.
3. Run any step with the new site name:

```bash
python -m runup.download_timestacks <SiteName>
```
