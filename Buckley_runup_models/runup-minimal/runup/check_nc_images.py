import argparse
import logging
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from scipy.signal import find_peaks
from typing import NamedTuple

from .pixel2runup import getSite


class RunupStats(NamedTuple):
    R: np.ndarray
    R2: float
    R50: float
    Rperiod: float
    minPeaksCriteria: int


def _get_z(ds, site):
    """Return elevation array from dataset using the site's vertical datum."""
    var = f'z_{site.verticalDatum}'
    if var in ds:
        return ds[var].data
    print(f'No z data found for datum {site.verticalDatum} in {ds.attrs.get("Name", "")}')
    return np.zeros_like(ds.x_local.data)


def _compute_runup_stats(ds, time) -> RunupStats:
    """Extract runup peaks and compute R2, R50, Rperiod from TWL timeseries."""
    minPeaksCriteria = 10
    empty            = RunupStats([], None, None, 0, minPeaksCriteria)

    if not hasattr(ds, 'twl'):
        return empty

    peaks, _ = find_peaks(ds.twl.data,
                          prominence=np.std(ds.twl.data) / 3, distance=3)
    R        = ds.twl.data[peaks]
    R2 = R50 = None

    if len(R) >= minPeaksCriteria and np.max(R) != np.min(R):
        counts, bin_edges = np.histogram(R, bins=20)
        bin_centers       = (bin_edges[:-1] + bin_edges[1:]) / 2
        cdf = np.cumsum(counts) / np.sum(counts)
        R2  = np.interp(0.98, cdf, bin_centers)
        R50 = np.interp(0.50, cdf, bin_centers)
    else:
        reason = (f'Too few peaks ({len(R)})'
                  if len(R) < minPeaksCriteria else 'min==max peaks')
        print(f'{reason} in: {ds.attrs.get("Name", "")}')

    Rperiod = max(time) / len(peaks) if len(peaks) else 0
    return RunupStats(R, R2, R50, Rperiod, minPeaksCriteria)


def _quality_codes(ds, R, R2, R50, Rperiod, minPeaksCriteria, min_pixel):
    """Return a list of 12 quality code flags (0=pass, 1=fail)."""
    code_list = [0] * 12

    if not hasattr(ds, 'Ri'):
        code_list[1] = 1
        return code_list

    Ri_idx = ds.Ri.data.astype(int)

    if sum(ds.Ri.data.astype(int) <= min_pixel) >= Ri_idx.shape[0] * 0.02:
        code_list[2] = 1
    if len(R) < minPeaksCriteria:
        code_list[3] = 1
    if len(R) and np.max(R) == np.min(R):
        code_list[4] = 1
    if Rperiod > 100:
        code_list[5] = 1
    if np.mean(ds.color.data) < 10:
        code_list[6] = 1
    if np.mean(ds.color.data) > 245:
        code_list[7] = 1

    x_ri = ds.x_local.data[Ri_idx]
    if (np.min(x_ri) < np.min(ds.x_local.data) + 1
            and np.max(x_ri) > np.max(ds.x_local.data) - 1):
        code_list[8] = 1

    # Image sharpness / brightness — only computed when Ri is present
    image_cv        = (cv2.cvtColor(ds.color.data, cv2.COLOR_RGB2GRAY)
                       if ds.color.data.ndim == 3 else ds.color.data)
    mean_brightness = np.mean(image_cv)
    laplacian_var   = cv2.Laplacian(image_cv, cv2.CV_64F).var()

    code_list[9]  = int(mean_brightness < 50)
    code_list[10] = int(mean_brightness > 200)
    code_list[11] = int(laplacian_var   < 120)

    return code_list


def _plot_timestack(ds, time, filename, site, R, R2, R50, Rperiod, code_list):
    """Create QA/QC figure for a single timestack netCDF."""
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(5, 1, (1, 4))

    ax1.pcolor(ds.x_local.data, time, ds.color.data, cmap='gray')
    if hasattr(ds, 'Ri'):
        ax1.plot(ds.x_local.data[ds.Ri.data.astype(int)], time, 'r', linewidth=0.2)

    ax1.set_ylabel('time (s)')
    ax1.set_xticks([])
    ax1.set_title(Path(filename).stem)
    ax1.text(0.8, -0.04, f'NumRunupPeaks={len(R)}', transform=ax1.transAxes)

    if hasattr(ds, 'TWLstats_TWL_2pct') and R2 is not None and R50 is not None:
        ax1.text(0.8, 0.02,
            f'TWL 2%: {ds.TWLstats_TWL_2pct.values:.2f}m\n'
            f'Tmean: {ds.TWLstats_Tmean.values:.2f}s\n'
            f'Tenergy: {ds.TWLstats_Tenergy.values:.2f}s\n'
            f'Rperiod: {Rperiod:.2f}s\nR2/R50: {R2/R50:.2f}',
            transform=ax1.transAxes, fontsize=8, verticalalignment='bottom',
            bbox=dict(facecolor='white', alpha=1))

    if not hasattr(ds, 'Ri'):
        ax1.text(1, 0.75, 'code:2-no runup', transform=ax1.transAxes)
    else:
        code_labels = {
             2: 'code:3-',
             3: 'code:4-few runup peaks (<10)',
             4: 'code:5-no R2 peaks',
             5: 'code:6-Rper>100',
             6: 'code:7-too dark',
             7: 'code:8-too bright',
             8: 'code:9-both edges',
             9: 'code:10-dark:shorescan',
            10: 'code:11-bright:shorescan',
            11: 'code:12-blurry:shorescan',
        }
        y_pos = 0.725
        for idx, label in code_labels.items():
            if code_list[idx]:
                ax1.text(1, y_pos, label, transform=ax1.transAxes)
            y_pos -= 0.025

    ax2 = fig.add_subplot(5, 1, 5)
    ax2.plot(ds.x_local.data, _get_z(ds, site), 'k')
    ax2.set_ylabel('elevation (m)')
    ax2.set_xlabel('x_local (m)')

    return fig, ax1


def check_nc_images(site_name, min_pixel=-99999):
    configDir = str(Path(__file__).resolve().parent.parent / 'config')
    site      = getSite(site_name, configDir)

    ncPath   = Path(site.netcdfDir)
    savePath = Path(site.figureDir)

    logDir = Path(__file__).resolve().parent.parent / 'logs'
    logDir.mkdir(exist_ok=True)
    logging.basicConfig(
        filename=str(logDir / f'{site.siteName}_qaqc.log'),
        level=logging.INFO, filemode='a', format='%(asctime)s, %(message)s'
    )

    ncFileList = sorted(ncPath.glob('*.nc'))
    if not ncFileList:
        print(f'No .nc files found in {ncPath}')
        return

    okayDir = savePath / 'runup_check_okay'
    maxDir  = savePath / 'runup_check_max'
    okayDir.mkdir(parents=True, exist_ok=True)
    maxDir.mkdir(parents=True,  exist_ok=True)

    existingNames = {f.stem for f in savePath.rglob('*.png')}

    for i, filepath in enumerate(ncFileList):
        filename = filepath.name
        print(f'{i + 1} of {len(ncFileList)}: {filepath}')

        if filepath.stem in existingNames:
            print(f'File already exists in output folder: {filename}')
            continue

        ds = xr.open_dataset(str(filepath))
        try:
            time      = (ds.time.data.astype(float) - ds.time[0].data.astype(float)) / 1e9
            stats     = _compute_runup_stats(ds, time)
            code_list = _quality_codes(
                ds, stats.R, stats.R2, stats.R50, stats.Rperiod,
                stats.minPeaksCriteria, min_pixel
            )
            fig, ax1  = _plot_timestack(
                ds, time, filename, site,
                stats.R, stats.R2, stats.R50, stats.Rperiod, code_list
            )
            outDir = maxDir  if sum(code_list) > 0 else okayDir
            label  = str(code_list) if sum(code_list) > 0 else 'code:1-pass'
            ax1.text(0.8, -0.018, label, transform=ax1.transAxes)
            plt.savefig(str(outDir / filepath.with_suffix('.png').name))
            logging.info(f'{filename}, {code_list}')
        except Exception as e:
            logging.info(f'{filename}, failed: {e}')
        finally:
            plt.close('all')
            ds.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QA/QC check on timestack netCDF files')
    parser.add_argument('site_name', help='Site name matching config file (e.g. Molokai_c2_runup03)')
    parser.add_argument('--min_pixel', type=int, default=-99999)
    args = parser.parse_args()
    check_nc_images(args.site_name, min_pixel=args.min_pixel)
