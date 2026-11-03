import ast
import sys
import logging
import argparse
import pytz
import utm
import numpy as np
import xarray as xr
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone, timedelta

from .pixel2runup import (getSite, getSoftMaxScores, createCalibDict, getUVVals,
                          findStackBreaks, loadMatStacks, loadTiffStacks,
                          localTransformPoints, getSurveyVals, computeRunupMapped2Topo,
                          replace_single_nan_with_mean, get_SegGym_runup_pixel_timeseries)

from .wave_stats import wave_stats

@dataclass
class TimestackData:
    T: np.ndarray
    UTM_E: np.ndarray
    UTM_N: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    xRaw: np.ndarray
    yRaw: np.ndarray
    xiRunup: np.ndarray
    yiRunup: np.ndarray
    ziRunup: np.ndarray
    RAW: np.ndarray
    U: np.ndarray
    V: np.ndarray
    av_prob_stack: np.ndarray
    xCamera: float
    yCamera: float
    zCamera: float
    localOriginLat: float
    localOriginLon: float
    sample_frequency_Hz: float
    sample_period_length_min: int
    sample_period_length_sec: int
    ncFilename: str
    fileDatetime: datetime
    onlyDate: str
    calibDict: dict
    history: str
    hasRi: bool = False
    Ri: np.ndarray = None
    Ri_0: np.ndarray = None
    Ri_down: np.ndarray = None
    Ri_up: np.ndarray = None
    twl: np.ndarray = None
    hasTWLstats: bool = False
    TWLstats: dict = None


def _get_cam_stack_pairs(site):
    """Extract (camNum, stackNum) arrays and unique pairs from site.timestackList."""
    camNumList, stackNumList = [], []
    for file in site.timestackList:
        parts = Path(file).name.split('.')
        camNumList.append(site.camNum if parts[7][-1].isalpha() else int(parts[7][-1]))
        if site.stackExtension != '.mat' or getattr(site, 'stackNumMap', None):
            try:
                sn = next(e.replace('runup', '') for e in parts if 'runup' in e)
            except StopIteration:
                sn = next(e.replace('ras',   '') for e in parts if 'ras'   in e)
            try:
                stackNumList.append(int(sn))
            except ValueError:
                stackNumList.append(-999)
        else:
            stackNumList.append(0)

    camNumList   = np.array(camNumList)
    stackNumList = np.array(stackNumList)
    validPairs   = np.column_stack((camNumList, stackNumList))[stackNumList >= 0]
    return camNumList, stackNumList, np.unique(validPairs, axis=0)


def _process_uv_xyz(site, U, V, XYZ, stackNum):
    """
    Trim and orient U, V, XYZ to the correct stack extent and cross-shore direction.

    Config flags:
      stackNumMap (dict)   - maps legacy stack numbers to sequential ones
      xyzPreTrimmed (bool) - skip XYZ trimming if already trimmed in the .mat file
    """
    stackBreaks = findStackBreaks(U, V)

    stackNumMap = getattr(site, 'stackNumMap', None)
    if stackNumMap:
        stackNum = stackNumMap.get(str(int(stackNum)), stackNum)

    idx  = int(stackNum) - 1 if stackNum > 0 else 0
    s, e = int(stackBreaks[idx, 0]), int(stackBreaks[idx, 1])
    U, V = U[s:e], V[s:e]

    if not getattr(site, 'xyzPreTrimmed', False):
        XYZ = XYZ[s:e, :]

    if np.mean(np.diff(XYZ[:, 0]) < 0) > 0.5:
        U   = np.flip(U)
        V   = np.flip(V)
        XYZ = np.flip(XYZ, axis=int(np.argmax(XYZ.shape)))
    return U, V, XYZ


def _correct_monotonic_time(T):
    """Ensure time array is strictly monotonic by fixing duplicate timestamps."""
    sample_frequency_Hz = 1 / np.mean(np.diff(T.astype('float64'))).round(0)
    previousTime = 0
    for j, t in enumerate(T):
        if t == previousTime:
            T[j] += 1 / sample_frequency_Hz
        previousTime = T[j]
    return T


def _get_camera_xy(yamlData, localOrigin):
    """Return camera (x, y) in local coordinates from parsed YAML data."""
    if yamlData['coordinate_system'] == 'xyz':
        return float(yamlData['x']), float(yamlData['y'])
    return localTransformPoints(
        localOrigin, float(yamlData['localAngle']), 1,
        float(yamlData['x']), float(yamlData['y'])
    )


def _should_reuse_survey(xRaw, xRaw_last, surveyDatetime, surveyDatetime_last,
                         zCamera, zCamera_last):
    """Return True if the topo survey result from the previous file can be reused."""
    return (
        len(xRaw_last) > 0
        and surveyDatetime   == surveyDatetime_last
        and zCamera_last     == zCamera
        and len(xRaw_last)   == len(xRaw)
        and all(xRaw_last    == xRaw)
    )


# ---------------------------------------------------------------------------
# Variable-level metadata table for _build_dataset.
# min_value / max_value are computed dynamically per variable.
# ---------------------------------------------------------------------------
_VAR_ATTRS = [
    ('x_utm', {
        'standard_name': 'projection_x_coordinate', 'units': 'meters', 'axis': 'X',
    }),
    ('y_utm', {
        'standard_name': 'projection_y_coordinate', 'units': 'meters', 'axis': 'Y',
    }),
    ('lon', {
        'standard_name': 'longitude', 'units': 'degrees_east', 'axis': 'X',
        'grid_mapping': 'crs_utm crs_wgs84 crs_local',
        'description':  'Location of timestack pixels described using longitude in decimal degrees.',
    }),
    ('lat', {
        'standard_name': 'latitude', 'units': 'degrees_north', 'axis': 'Y',
        'grid_mapping': 'crs_utm crs_wgs84 crs_local',
        'description':  'Location of timestack pixels described using latitude in decimal degrees.',
    }),
    ('x_local', {
        'standard_name': 'projection_x_coordinate', 'units': 'meters', 'axis': 'X',
        'reference_frame': 'local', '_FillValue': np.nan,
        'description': 'x-coordinate in right-handed Cartesian system, positive x directed offshore.',
    }),
    ('y_local', {
        'standard_name': 'projection_y_coordinate', 'units': 'meters', 'axis': 'Y',
        'reference_frame': 'local', '_FillValue': np.nan,
        'description': 'y-coordinate in right-handed Cartesian system, y directed alongshore.',
    }),
    ('x_local_z0', {
        'standard_name': 'projection_x_coordinate_z0', 'units': 'meters', 'axis': 'X',
        'reference_frame': 'local', '_FillValue': np.nan,
        'description': 'x-coordinate with z = 0, positive x directed offshore.',
    }),
    ('y_local_z0', {
        'standard_name': 'projection_y_coordinate_z0', 'units': 'meters', 'axis': 'Y',
        'reference_frame': 'local', '_FillValue': np.nan,
        'description': 'y-coordinate with z = 0, y directed alongshore.',
    }),
    ('u', {
        'standard_name': 'image_x_coordinate', 'units': 'pixels', 'axis': 'X',
        'long_name': 'pixel coordinate along the horizontal image axis',
        'grid_mapping': 'crs_utm crs_wgs84 crs_local',
    }),
    ('v', {
        'standard_name': 'image_y_coordinate', 'units': 'pixels', 'axis': 'Y',
        'long_name': 'pixel coordinate along the vertical image axis',
        'grid_mapping': 'crs_utm crs_wgs84 crs_local',
    }),
    ('color', {
        'standard_name': 'pixel_intensity', 'units': '1', 'valid_range': [0, 255],
        'coordinates':   'time x_utm', 'grid_mapping': 'crs_utm crs_wgs84 crs_local',
        'description':   '8-bit image color values of the timestack.',
    }),
    ('softmax', {
        'standard_name': 'probability', 'units': '1', 'valid_range': [0, 1],
        'long_name':     'softmax scores',
        'description':   'Per-pixel probability of being classified as water (negative) or land (positive).',
        'coordinates':   'time x_utm', 'grid_mapping': 'crs_utm crs_wgs84 crs_local',
    }),
]


def _build_dataset(site, d: TimestackData) -> xr.Dataset:
    """Build and return a fully attributed xarray Dataset from a TimestackData object."""
    color_dims = ('time', 'x', 'band') if d.RAW.ndim == 3 else ('time', 'x')
    coords = {
        'time': ('time', d.T),
        'x':    ('x',    np.arange(len(d.UTM_E))),
        **({'band': ('band', np.arange(3))} if d.RAW.ndim == 3 else {}),
    }

    data_vars = {
        'x_utm':       xr.DataArray(d.UTM_E,        dims='x'),
        'y_utm':       xr.DataArray(d.UTM_N,        dims='x'),
        'lon':         xr.DataArray(d.lon,           dims='x'),
        'lat':         xr.DataArray(d.lat,           dims='x'),
        'x_local_z0':  xr.DataArray(d.xRaw,         dims='x'),
        'y_local_z0':  xr.DataArray(d.yRaw,         dims='x'),
        'x_local':     xr.DataArray(d.xiRunup,       dims='x'),
        'y_local':     xr.DataArray(d.yiRunup,       dims='x'),
        f'z_{site.verticalDatum}': xr.DataArray(d.ziRunup, dims='x'),
        'color':       xr.DataArray(d.RAW,           dims=color_dims),
        'u':           xr.DataArray(d.U,             dims='x'),
        'v':           xr.DataArray(d.V,             dims='x'),
        'softmax':     xr.DataArray(d.av_prob_stack, dims=('time', 'softmax')),
        'x_camera_local': xr.DataArray(d.xCamera,
                              attrs={'long_name': 'Camera local x coordinate', 'units': 'meters'}),
        'y_camera_local': xr.DataArray(d.yCamera,
                              attrs={'long_name': 'Camera local y coordinate', 'units': 'meters'}),
        'z_camera_local': xr.DataArray(d.zCamera,
                              attrs={'long_name': 'Camera elevation',           'units': 'meters'}),
    }

    if d.hasRi:
        data_vars.update({
            'Ri':      xr.DataArray(d.Ri,      dims='time',
                           attrs={'long_name': 'Optimized runup contour'}),
            'Ri_0':    xr.DataArray(d.Ri_0,    dims='time',
                           attrs={'long_name': 'Runup contour at softmax=0'}),
            'Ri_down': xr.DataArray(d.Ri_down, dims='time',
                           attrs={'long_name': 'Runup contour at rundown threshold'}),
            'Ri_up':   xr.DataArray(d.Ri_up,   dims='time',
                           attrs={'long_name': 'Runup contour at runup threshold'}),
            'twl':     xr.DataArray(d.twl,     dims='time',
                           attrs={'long_name': 'Total water level'}),
        })

    if d.hasTWLstats:
        for key, value in d.TWLstats.items():
            if not isinstance(value, dict):
                continue
            arr = np.array(value['data'])
            if key in ('f', 'Sp'):
                dims = ('frequency',)
            elif key == 'freq_bins':
                dims = ('freq_bins', 'bounds')
            elif arr.ndim > 0 and len(arr) == 3:
                dims = ('freq_bands',)
            else:
                dims = None
            data_vars[f'TWLstats_{key}'] = xr.DataArray(
                data=value['data'], dims=dims,
                attrs={'long_name': value['long_name'], 'units': value['units']}
            )

    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs={
        'Name':        d.ncFilename,
        'Conventions': 'CF-1.6',
        'title':       f"{site.siteLocation} {d.fileDatetime} UTC: timestack image",
        'date':        d.onlyDate,
        'year':        d.fileDatetime.year,
        'institution': 'U.S. Geological Survey',
        'program':     'Coastal-Marine Hazards and Resources',
        'project':     'Next Generation Total Water Level and Coastal Change Forecasts',
        'contributors': site.contributors,
        'references':   site.references,
        'site':         site.siteLocation,
        'description': (
            f'pixel intensity timestack sampled at {d.sample_frequency_Hz} Hz for '
            f'{d.sample_period_length_min} minutes at {site.siteLocation}. '
            f'Collection began at {d.fileDatetime} UTC'
        ),
        'data_origin':       site.dataOrigin,
        'local_timezone':    site.timezone,
        'source':            'Mounted camera image capture',
        'coord_system':      'UTM',
        'utm_zone':          site.utmZone,
        'sample_period_length': f'{d.sample_period_length_min} minutes',
        'cam_make':          site.camMake,
        'cam_model':         site.camModel,
        'cam_lens':          site.camLens,
        'data_type':         'time series',
        'calibration_parameters': (
            "The following nested dictionary contains the camera's intrinsic and extrinsic "
            f"calibration parameters computed using the Brown distortion model "
            f"(Brown, Duane C. May 1966)/n{d.calibDict}"
        ),
        'history': d.history,
    })

    ds['crs_utm']   = xr.DataArray(attrs={
        'grid_mapping_name':               'transverse_mercator',
        'scale_factor_at_central_meridian': 0.999600,
        'longitude_of_central_meridian':   -81.0,
        'latitude_of_projection_origin':    0.0,
        'false_easting':                   500000.0,
        'false_northing':                   0.0,
    })
    ds['crs_wgs84'] = xr.DataArray(attrs={'grid_mapping_name': 'latitude_longitude'})
    ds['crs_local'] = xr.DataArray(attrs={
        'grid_mapping_name':              'vertical_perspective',
        'latitude_of_projection_origin':  d.localOriginLat,
        'longitude_of_projection_origin': d.localOriginLon,
        'perspective_point_height':       d.zCamera,
    })

    ds['time'].attrs.update({
        'standard_name':          'time',
        'long_name':              'Time of image collection',
        'axis':                   'T',
        'units':                  'seconds since 1970-01-01 00:00:00 UTC',
        'time_zone':              'UTC',
        'description':            'Times that pixels were sampled to create the timestack.',
        'sample_freq':            f'{d.sample_frequency_Hz} Hertz',
        'sample_length_interval': f'{d.sample_period_length_sec} seconds',
        'min_value':              np.around(d.T.min(), 2),
        'max_value':              np.around(d.T.max(), 2),
    })

    # Apply metadata table; auto-compute min/max per variable
    for var, extra in _VAR_ATTRS:
        if var not in ds:
            continue
        vals     = ds[var].values
        finite   = vals[np.isfinite(vals)] if np.issubdtype(vals.dtype, np.floating) else vals
        attrs    = dict(extra)
        decimals = 5 if var in ('lon', 'lat') else 2
        if finite.size:
            attrs['min_value'] = np.around(finite.min(), decimals)
            attrs['max_value'] = np.around(finite.max(), decimals)
        ds[var].attrs.update(attrs)

    # z_<datum> needs site-specific attrs not in the generic table
    zvar = f'z_{site.verticalDatum}'
    ds[zvar].attrs.update({
        'standard_name':  'height',
        'units':          'meters',
        'axis':           'Z',
        'vertical_datum': site.verticalDatum,
        'long_name':      'elevation',
        'description':    f'Elevation in {site.verticalDatum} of timestack pixels at {site.siteLocation}.',
        'datum':          site.verticalDatum_description,
        'min_value':      np.around(np.nanmin(d.ziRunup), 2),
        'max_value':      np.around(np.nanmax(d.ziRunup), 2),
    })

    return ds


def _process_one_timestack(
    site, file, camNum, stackNum,
    xRaw_last, surveyDatetime_last, zCamera_last,
    xiRunup, yiRunup, ziRunup, XYZi,
):
    """
    Process a single timestack file into an xarray Dataset.

    Handles all per-file I/O, coordinate transforms, runup extraction, and
    wave statistics. Survey computation is skipped when inputs match the
    previous file (xRaw, surveyDatetime, zCamera unchanged), with the cached
    interpolated XYZ passed in and returned unchanged.

    Args:
        site                - Site object
        file                - path to timestack file (str or Path)
        camNum              - camera number (int)
        stackNum            - stack number (int)
        xRaw_last           - xRaw from previous file, for survey cache check
        surveyDatetime_last - survey datetime from previous file
        zCamera_last        - camera z from previous file
        xiRunup, yiRunup, ziRunup - cached interpolated coords from previous file
        XYZi                - cached interpolated XYZ array from previous file

    Returns:
        (ds, datasetLoc, xRaw, surveyDatetime, zCamera, xiRunup, yiRunup, ziRunup, XYZi)
        or None if the file should be skipped.
    """
    filepath = Path(file)
    filename = filepath.name
    parts    = filename.split('.')

    fileDatetime = datetime.fromtimestamp(int(parts[0]), tz=pytz.utc)
    onlyDate     = fileDatetime.strftime('%Y-%m-%d')

    if (fileDatetime < site.startDate) or (fileDatetime > site.endDate + timedelta(days=1)):
        return None

    try:
        av_prob_stack = getSoftMaxScores(site.segDir, filename)
    except Exception:
        logging.info('Skipping file: no segmentation data')
        return None

    skipCams = getattr(site, 'skipCams', [])
    if int(camNum) in skipCams:
        logging.info(f'Skipping camera {camNum} (in skipCams config)')
        return None

    ncFilename = filepath.stem + '.nc'
    datasetLoc = Path(site.netcdfDir) / ncFilename

    if site.stackExtension == '.mat':
        XYZ, T, RAW = loadMatStacks(file)
        print(f'loadMatStacks RAW= {RAW.shape}')
    else:
        try:
            XYZ, T, RAW = loadTiffStacks(site, file, camNum,
                                          sample_frequency_Hz=site.tiff_sample_frequency_Hz)
        except ValueError:
            logging.info('Failed: loadTiffStacks failed')
            return None

    if np.isnan(XYZ).any() or np.isnan(T).any() or np.isnan(RAW).any():
        logging.info('Failed: nan found in timestack')
        return None

    history = (f"{datetime.now(timezone.utc).isoformat()} Using Python {sys.version}, "
               f"xarray {xr.__version__}, NumPy {np.__version__}")

    T = _correct_monotonic_time(T)
    history += f'\n {datetime.now(timezone.utc).isoformat()} correcting T to be strictly monotonic'

    time_sec             = T.astype('float64')
    sample_frequency_Hz  = (1 / np.mean(np.diff(time_sec))).round(0)
    sample_period_length_min = int(np.round((time_sec[-1] - time_sec[0]) / 60))
    sample_period_length_sec = int(np.round(time_sec[-1] - time_sec[0])) + 1

    calibDict = createCalibDict(site, fileDatetime=fileDatetime, camNum=camNum)
    yamlData  = ast.literal_eval(calibDict['data'])

    U, V = getUVVals(site, int(parts[0]), camNum=camNum)

    if site.stackExtension != '.mat' or getattr(site, 'stackNumMap', None):
        U, V, XYZ = _process_uv_xyz(site, U, V, XYZ, stackNum)

    xRaw, yRaw, zRaw = XYZ[:, 0], XYZ[:, 1], XYZ[:, 2]

    if RAW.shape[0] != T.shape[0]:
        logging.info('RAW, T arrays are not the same length!')
    if RAW.shape[1] != U.shape[0]:
        logging.info('RAW, U, and V arrays are not the same length!')
    if RAW.ndim == 3 and RAW.shape[2] == 4:
        RAW = RAW[:, :, :3]
        history += f'\n {datetime.now(timezone.utc).isoformat()} slicing color dimension to length 3'

    localOrigin = np.array([float(yamlData['x_origin']), float(yamlData['y_origin'])])
    localOriginLat, localOriginLon = utm.to_latlon(
        float(yamlData['x_origin']), float(yamlData['y_origin']),
        int(site.utmZone[:2]), site.utmZone[2:3]
    )
    UTM_E, UTM_N = localTransformPoints(
        localOrigin, float(yamlData['localAngle']), 0, xRaw, yRaw)
    history += f'\n {datetime.now(timezone.utc).isoformat()} converting UTM xy from local coordinates'

    xSurvey, ySurvey, zSurvey, dateObjZ = getSurveyVals(int(parts[0]), site)
    surveyDatetime   = datetime.strptime(dateObjZ, '%b %d, %Y').replace(tzinfo=timezone.utc)
    xCamera, yCamera = _get_camera_xy(yamlData, localOrigin)
    zCamera          = float(yamlData['z'])

    if not _should_reuse_survey(xRaw, xRaw_last, surveyDatetime,
                                surveyDatetime_last, zCamera, zCamera_last):
        print('Running computeRunupMapped2Topo')
        xiRunup, yiRunup, ziRunup = computeRunupMapped2Topo(
            xRaw, yRaw, zRaw,
            xSurvey.flatten(), ySurvey.flatten(), zSurvey.flatten(),
            xCamera, yCamera, zCamera
        )
        xiRunup = replace_single_nan_with_mean(xiRunup)
        yiRunup = replace_single_nan_with_mean(yiRunup)
        ziRunup = replace_single_nan_with_mean(ziRunup)
        XYZi    = np.stack((xiRunup, yiRunup, ziRunup), axis=0).T.squeeze()
        print(f'{onlyDate}: Topo {datetime.strptime(dateObjZ, "%b %d, %Y").strftime("%Y%m%d")}')
        history += (f'\n {datetime.now(timezone.utc).isoformat()} '
                    f'generating interpolated local xyz from survey')

    hasRi = False
    Ri = Ri_0 = Ri_down = Ri_up = twl = None
    try:
        Ri, Ri_0, Ri_down, Ri_up = get_SegGym_runup_pixel_timeseries(
            av_prob_stack, -1.5, 0, (0, 2))
        hasRi = True
        twl   = XYZi[np.round(Ri).astype(int), 2]
        history += f'\n {datetime.now(timezone.utc).isoformat()} generating twl contours from softmax'
        history += f'\n {datetime.now(timezone.utc).isoformat()} calculating wave-driven runup component'
    except Exception:
        logging.info('Failed: get_SegGym_runup_pixel_timeseries failed')

    hasTWLstats = False
    TWLstats    = None
    if hasRi:
        nperseg = 1 << (int(np.floor(len(time_sec) / 8)) - 1).bit_length()
        try:
            TWLstats = wave_stats(twl, 1 / sample_frequency_Hz,
                                  IG_divide=1/26, nperseg=nperseg)
            if np.isnan(TWLstats['Hrms']['data'][0]):
                logging.info(
                    f"Warning: wave_stats has nan. "
                    f"NaN count: {np.sum(np.isnan(twl))}/{len(twl)}")
            hasTWLstats = True
        except Exception:
            logging.info(
                f"Failed: wave_stats failed. "
                f"NaN count: {np.sum(np.isnan(twl))}/{len(twl)}")

    lat, lon = utm.to_latlon(UTM_E, UTM_N, int(site.utmZone[:2]), site.utmZone[2:3])

    d = TimestackData(
        T=T, UTM_E=UTM_E, UTM_N=UTM_N, lat=lat, lon=lon,
        xRaw=xRaw, yRaw=yRaw, xiRunup=xiRunup, yiRunup=yiRunup, ziRunup=ziRunup,
        RAW=RAW, U=U, V=V, av_prob_stack=av_prob_stack,
        xCamera=xCamera, yCamera=yCamera, zCamera=zCamera,
        localOriginLat=localOriginLat, localOriginLon=localOriginLon,
        sample_frequency_Hz=sample_frequency_Hz,
        sample_period_length_min=sample_period_length_min,
        sample_period_length_sec=sample_period_length_sec,
        ncFilename=ncFilename, fileDatetime=fileDatetime, onlyDate=onlyDate,
        calibDict=calibDict, history=history,
        hasRi=hasRi, Ri=Ri, Ri_0=Ri_0, Ri_down=Ri_down, Ri_up=Ri_up, twl=twl,
        hasTWLstats=hasTWLstats, TWLstats=TWLstats,
    )

    return _build_dataset(site, d), datasetLoc, xRaw, surveyDatetime, zCamera, \
           xiRunup, yiRunup, ziRunup, XYZi


def timestack2netcdf(site_name):
    print('start:', datetime.now())
    configDir = str(Path(__file__).resolve().parent.parent / 'config')
    site      = getSite(site_name, configDir)
    print(site.startDate, site.endDate)
    site.gatherTimestacks()

    # Filter out files that already have a NetCDF output
    if not site.writeOverNC:
        done = {Path(f).stem for f in Path(site.netcdfDir).glob('*.nc')}
        site.timestackList = [f for f in site.timestackList if Path(f).stem not in done]

    logDir = Path(__file__).resolve().parent.parent / 'logs'
    logDir.mkdir(exist_ok=True)
    logging.basicConfig(
        filename=str(logDir / f'{site.siteName}_timestack2netcdf.log'),
        level=logging.INFO, filemode='a',
        format='%(asctime)s -- %(levelname)s -- %(message)s'
    )
    logging.info(f'---writing timestacks to NetCDF for {site.siteName}---')

    camNumList, stackNumList, uniquePairs = _get_cam_stack_pairs(site)

    for camNum, stackNum in uniquePairs:
        print(f'camera {camNum} - stacknum {stackNum}')
        id_current = (np.arange(len(site.timestackList))
                      if uniquePairs.shape[0] == 1
                      else np.where((camNumList == camNum) & (stackNumList == stackNum))[0])

        xRaw_last = np.array([])
        surveyDatetime_last = zCamera_last = None
        xiRunup = yiRunup = ziRunup = XYZi = None

        for i, file in enumerate([site.timestackList[j] for j in id_current.astype(int)]):
            print(i)
            logging.info(f'Loading timestack {i + 1} of {len(site.timestackList)}: {file}')

            result = _process_one_timestack(
                site, file, camNum, stackNum,
                xRaw_last, surveyDatetime_last, zCamera_last,
                xiRunup, yiRunup, ziRunup, XYZi,
            )
            if result is None:
                continue

            ds, datasetLoc, xRaw_last, surveyDatetime_last, zCamera_last, \
                xiRunup, yiRunup, ziRunup, XYZi = result

            ds.to_netcdf(str(datasetLoc))
            logging.info('Successful')

    logging.info('--- Done creating netCDF files ---')
    print('end:', datetime.now())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert timestacks to netCDF')
    parser.add_argument('site_name', help='Site name matching config file (e.g. Molokai_c2_runup03)')
    timestack2netcdf(parser.parse_args().site_name)
