# functions for the pixel2runup (timestack to NetCDF) scripts. Sorted alphabetically.

##### IMPORT #####
from pathlib import Path
import utm, calendar
import numpy as np
import numpy.matlib
import scipy
import scipy.signal
import scipy.interpolate as interp
import cv2
import os
import pytz
import json
import yaml
import matplotlib.pyplot as plt
import xarray as xr
from scipy.stats import trim_mean
from datetime import datetime, timezone, date
import pandas as pd


##### CLASSES #####
class Site:
    """
    Class to hold info related to each CoastCam site. The data and metadata stored in each instance
    will be standardized to easily create NetCDF files from them.
    Attributes:
        siteName (string)   - name of the site (matches config filename)
        nName (string)      - site fragment used in timestack filenames
        saveName (string)   - short name used to match YAML calibration files
        netcdfDir (Path)    - output directory for NetCDF files
        rawDirTop (Path)    - top-level directory for raw Argus archive
        topoDir (Path)      - directory containing topo survey files
        uvDir (Path)        - directory containing UV data
        references (string) - citations
        contributors (string)
        siteLocation (string)
        dataOrigin (string)
        camMake / camModel / camLens (string)
        timezone (string)   - local timezone
        utmZone (string)    - UTM zone (number + letter)
        timestackList (list) - filepaths to raw timestacks
    """

    def __init__(self, siteName, nName, saveName=None, configFile=''):
        self.siteName   = siteName
        self.nName      = nName
        self.saveName   = saveName if saveName is not None else ''
        self.configFile = configFile

    def gatherTimestacks(self):
        """Generate and store the list of timestacks to be processed."""
        if hasattr(self, 'rawDirTop') and self.rawDirTop != '':
            print(f'Using files from rawDirTop: {self.rawDirTop}')
            timestackFromRawDirTop(self)
        else:
            print(f'Using files from stackDir: {self.stackDir}')
            match_name = self.saveName if self.saveName == 'madbeach' else self.nName
            self.timestackList = [
                str(f) for f in Path(self.stackDir).glob(f'*{self.stackExtension}')
                if match_name in f.name.lower()
                and f.name.split('.')[0].isdigit()
            ]

    def loadJson(self):
        """
        Load attributes from json config file.

        exampleStackName is processed first to derive stackExtension, nName,
        camNum, and rNumber. Any of those keys appearing explicitly in the config
        after exampleStackName will override the derived values.
        """
        _dir_keys = {
            'netcdfDir', 'figureDir', 'topoDir', 'uvTopDir', 'xyzTopDir',
            'rawDirTop', 'stackDir', 'segDir', 'modelDir'
        }
        with open(self.configFile) as f:
            jsonDict = json.load(f)
        for category in jsonDict:
            print(f"Loading attributes for category: {category}: {list(jsonDict[category].keys())}")
            for key, val in jsonDict[category].items():
                if key in ('startDate', 'endDate'):
                    datestr = val + '-00-00'
                    setattr(self, key + 'Str', datestr)
                    setattr(self, key,
                            datetime.strptime(datestr, '%Y-%m-%d-%H-%M').replace(tzinfo=pytz.utc))
                elif key in _dir_keys and val:
                    setattr(self, key,
                            Path(val.replace('/', os.sep).replace('\\', os.sep)))
                elif key == 'exampleStackName' and val:
                    setattr(self, key, val)
                    parts = Path(val).name.split('.')
                    # Derive stackExtension, nName, camNum, rNumber from example filename.
                    # Explicit values in the config after this key will override these.
                    setattr(self, 'stackExtension', Path(val).suffix)
                    if len(parts) >= 7:
                        setattr(self, 'nName', parts[6])
                    if len(parts) >= 8:
                        cam = parts[7]
                        if cam != 'cx' and cam[0] == 'c' and cam[1:].isdigit():
                            setattr(self, 'camNum', int(cam[1:]))
                    if len(parts) >= 9 and 'runup' in parts[8]:
                        setattr(self, 'rNumber', parts[8].replace('runup', ''))
                else:
                    setattr(self, key, val)


##### INTERNAL HELPERS #####

def _load_pix_mat(site, attr, epochTime, filename_fn):
    """
    Find the closest pix_* directory to epochTime, set site.<attr> to that
    directory, and return the loaded .mat struct.

    Supports both structured (xyzTopDir/pix_YYYYMMDD/) and flat (xyzTopDir/)
    layouts — if no pix_* subdirectories exist, loads directly from xyzTopDir.

    Args:
        site        - Site object (must have xyzTopDir)
        attr        - site attribute to update ('xyzDir' or 'uvDir')
        epochTime   - epoch timestamp (int)
        filename_fn - callable(site) -> str, returns the .mat filename to load
    Returns:
        matStruct from scipy.io.loadmat
    """
    topDir    = Path(site.xyzTopDir)
    pixelDirs = list(topDir.glob('pix_*'))

    if pixelDirs:
        fileDatetime  = datetime.fromtimestamp(epochTime)
        pixelDateList = getPixDirDateList([str(p) for p in pixelDirs])
        ind = getClosestPreviousDateIndex(pixelDateList, fileDatetime, useFirstIfNoneFound=True)
        setattr(site, attr, pixelDirs[ind])
    else:
        setattr(site, attr, topDir)

    matPath = Path(getattr(site, attr)) / filename_fn(site)
    return scipy.io.loadmat(str(matPath), squeeze_me=True, struct_as_record=False)


def _get_calib_yaml(site, suffix, fileDatetime=None, camNum=None):
    """
    Load an EO / IO / metadata YAML calibration file for a site.

    Args:
        site         - Site object
        suffix       - '_EO', '_IO', or '_metadata'
        fileDatetime - if given, find the closest-date YAML; else use undated file
        camNum       - camera number (int or None)
    Returns:
        (data_dict, yaml_path)
    """
    if camNum is not None:
        hasCamNum = True
    else:
        hasCamNum = hasattr(site, 'camNum')
        if hasCamNum:
            camNum = site.camNum

    yamlDir = Path(__file__).resolve().parent.parent / 'calibration'

    if fileDatetime is not None:
        pattern      = (f'{site.saveName}_c{camNum}*{suffix}.yaml' if hasCamNum
                        else f'{site.saveName}*{suffix}.yaml')
        yamlList     = list(yamlDir.glob(pattern))
        yamlDateList = uniqueYamlDates(yamlList)
        ind          = getClosestPreviousDateIndex(
                           yamlDateList, fileDatetime, useFirstIfNoneFound=True)
        closestDate  = datetime.strftime(yamlDateList[ind], '%Y%m%d')
        stem         = (f'{site.saveName}_c{camNum}_{closestDate}{suffix}' if hasCamNum
                        else f'{site.saveName}_{closestDate}{suffix}')
    else:
        stem = (f'{site.saveName}_c{camNum}{suffix}' if hasCamNum
                else f'{site.saveName}{suffix}')

    yamlPath = yamlDir / f'{stem}.yaml'
    return yaml2dict(yamlPath), yamlPath


##### PUBLIC FUNCTIONS #####

def initalizeSite(nameSnakeCase, configDir):
    configFile = Path(configDir) / f'{nameSnakeCase}_config.json'
    with open(configFile) as f:
        config   = json.load(f)
    meta     = config['metadata']
    saveName = meta.get('saveName', nameSnakeCase.lower())
    # nName may be overridden by exampleStackName in loadJson
    nName    = meta.get('nName', saveName.lower())
    return Site(nameSnakeCase, nName, saveName, configFile=configFile)


def getSite(nameSnakeCase, configDir):
    """Initialize and load a single site by name.
    Args:
        nameSnakeCase (str): Site name matching the config filename (e.g. 'Molokai_c2_runup03')
        configDir (str): Path to the config directory
    Returns:
        site (Site): Fully initialized Site object with JSON config loaded
    """
    site = initalizeSite(nameSnakeCase, configDir)
    site.loadJson()
    return site


def replace_single_nan_with_mean(arr):
    """Replace NaN values in a 1D array with the mean of the neighbours."""
    for i in np.where(np.isnan(arr))[0]:
        if i == 0:
            arr[i] = arr[i+1] - (arr[i+2] - arr[i+1])
        elif i == len(arr) - 1:
            arr[i] = arr[i-1] + (arr[i-2] - arr[i-1])
        else:
            arr[i] = np.mean(arr[[i-1, i+1]])
    return arr

# interpTopoCrossShoreNearest moved to extra_functions.py

def computeRunupMapped2Topo(
    xRunup, yRunup, zRunup, xTopo, yTopo, zTopo, xCamera, yCamera, zCamera
):
    """
    Convert horizontal runup (x,y) to vertical runup (z) using lidar data.

    Projects runup pixel positions onto the survey surface by converting both
    sets of points to camera-relative azimuth/elevation angles, then
    interpolating survey (x,y,z) at the pixel angles.

    Inputs:
      xRunup, yRunup, zRunup  - digitized horizontal runup (np arrays)
      xTopo, yTopo, zTopo     - survey topography data (np arrays)
      xCamera, yCamera, zCamera - camera position (floats)
    Outputs:
      xiRunup, yiRunup, ziRunup - runup coordinates mapped to survey surface
    """
    c0   = [xCamera, yCamera, zCamera]
    good = np.argwhere(~np.isnan(zTopo))
    xTopo, yTopo, zTopo = xTopo[good].squeeze(), yTopo[good].squeeze(), zTopo[good].squeeze()

    def _to_angles(x, y, z):
        dx, dy, dz = x - c0[0], y - c0[1], z - c0[2]
        hypotxy    = np.hypot(dx, dy)
        return np.arctan2(dy, dx), np.arctan2(dz, hypotxy)

    az_topo, el_topo = _to_angles(xTopo, yTopo, zTopo)
    coords = np.array([az_topo, el_topo]).T
    Fx = interp.LinearNDInterpolator(coords, xTopo)
    Fy = interp.LinearNDInterpolator(coords, yTopo)
    Fz = interp.LinearNDInterpolator(coords, zTopo)

    az_px, el_px = _to_angles(xRunup, yRunup, zRunup)
    return Fx(az_px, el_px), Fy(az_px, el_px), Fz(az_px, el_px)


def replace_non_monotonic_with_nan(xiRunup, trim_percentage=0.1):
    """
    Replace non-monotonically increasing or decreasing values with np.nan.

    Args:
        xiRunup (np.array): 1D float64 array.
        trim_percentage (float): Trim fraction for mean. Default 0.1.
    Returns:
        np.array with non-monotonic values replaced by np.nan.
    """
    if len(xiRunup) <= 1:
        return xiRunup.copy()

    xiRunup  = np.copy(xiRunup)
    nan_mask = np.isnan(xiRunup)
    non_nan  = xiRunup[~nan_mask]

    if len(non_nan) <= 1:
        return xiRunup

    diffs = np.diff(non_nan)
    if diffs.size == 0:
        return xiRunup

    mean_diff = trim_mean(diffs, trim_percentage)
    if mean_diff > 0:
        increasing = True
    elif mean_diff < 0:
        increasing = False
    else:
        nz = np.where(diffs != 0)[0]
        if nz.size == 0:
            return xiRunup
        increasing = diffs[nz[0]] > 0

    bad = np.where(diffs <= 0 if increasing else diffs >= 0)[0]
    xiRunup[np.where(~nan_mask)[0][bad + 1]] = np.nan
    return xiRunup


def createCalibDict(site, fileDatetime=None, camNum=None):
    """
    Build a calibration dictionary from EO, IO, metadata, and localOrigin YAMLs.

    Inputs:
        site         - Site object
        fileDatetime - Optional datetime for date-matched YAML selection
        camNum       - Optional camera number
    Outputs:
        calibDict    - dict with 'Note', 'data', and 'descriptions' keys
    """
    calibDict = {
        'Note': (
            'The following nested dictionary contains the cameras intrinsic and extrinsic '
            'calibration parameters. Intrinsic camera parameters use the Brown distortion model '
            '(Brown, D.C., 1971, "Close-Range Camera Calibration", Photogrammetric Engineering.), '
            'using the Camera Calibration Toolbox for Matlab by Jean-Yves Bouguet '
            '(https://doi.org/10.22002/D1.20164). Extrinsic calibration parameters were computed '
            'using the CIRN Quantitative Coastal Imaging Toolbox (Bruder, B. L. and Brodie, K. L., '
            '2020, "CIRN Quantitative Coastal Imaging Toolbox", SoftwareX, 12 (10052), '
            'https://doi.org/10.1016/j.softx.2020.100582.).'
        )
    }

    kwargs  = dict(fileDatetime=fileDatetime, camNum=camNum)
    sources = [
        _get_calib_yaml(site, '_EO',       **kwargs),
        _get_calib_yaml(site, '_IO',        **kwargs),
        _get_calib_yaml(site, '_metadata',  **kwargs),
        getLocalOrigin(site),
    ]

    data_fields, comment_fields = {}, {}
    for data, path in sources:
        data_fields.update({k: str(v) for k, v in data.items()})
        comment_fields.update(readYAMLcomments(path))

    calibDict['data']         = str(data_fields)
    calibDict['descriptions'] = str(comment_fields)
    return calibDict


# createNcFilename, datumFromDatetime, getPressure, extractBarometricPressure,
# extractMeanPressure, subtractPressureTS, findDunexUVFile, findRightBound,
# findLeftBound moved to extra_functions.py


def first_greater_than(A, dim, val):
    """Return the first index in array A greater than val along dimension dim."""
    return np.argmax(A > val, axis=dim)


# getCamEastingNorthing, local2latlon moved to extra_functions.py


def getClosestPreviousDateIndex(datetimeList, currentDatetime, useFirstIfNoneFound=False):
    """
    Return the index of the closest datetime in datetimeList that is <= currentDatetime.
    All datetimes must be timezone-naive.
    """
    closestIndex         = None
    lowestDateDifference = None
    currentDatetime      = currentDatetime.replace(tzinfo=None)
    for k, dNum in enumerate(datetimeList):
        if dNum <= currentDatetime:
            diff = abs(currentDatetime - dNum)
            if closestIndex is None or diff < lowestDateDifference:
                closestIndex         = k
                lowestDateDifference = diff
    if useFirstIfNoneFound and closestIndex is None:
        closestIndex = 0
    return closestIndex


# Public wrappers kept for backward compatibility ----------------------------

def getExtrinsics(site, fileDatetime=None, camNum=None):
    """Get extrinsic calibration dict and file path for a site."""
    return _get_calib_yaml(site, '_EO', fileDatetime=fileDatetime, camNum=camNum)

def getIntrinsics(site, fileDatetime=None, camNum=None):
    """Get intrinsic calibration dict and file path for a site."""
    return _get_calib_yaml(site, '_IO', fileDatetime=fileDatetime, camNum=camNum)

def getMetadata(site, fileDatetime=None, camNum=None):
    """Get metadata calibration dict and file path for a site."""
    return _get_calib_yaml(site, '_metadata', fileDatetime=fileDatetime, camNum=camNum)

# ---------------------------------------------------------------------------

def getLocalOrigin(site):
    """Return (localOrigin dict, localOriginFile Path) for a site."""
    yamlDir         = Path(__file__).resolve().parent.parent / 'calibration'
    localOriginFile = yamlDir / f'{site.saveName}_localOrigin.yaml'
    return yaml2dict(localOriginFile), localOriginFile


def getPixDirDateList(pixDirList, egFileString='pix_yyyymmdd'):
    """Return datetimes parsed from pix_YYYYMMDD folder names."""
    prefix   = egFileString.split('_')[0]
    dateList = []
    for path in pixDirList:
        name = Path(path).name
        if name.startswith(prefix) and len(name) == len(egFileString):
            dateList.append(datetime.strptime(name.split('_')[1], '%Y%m%d'))
    return dateList


def get_SegGym_runup_pixel_timeseries(softmax, rundown_val, runup_val, buffer):
    """
    Extract a runup contour timeseries from softmax scores using a weighted
    blend of runup and rundown phase masks.

    Returns:
        Ri, Ri_0, Ri_down, Ri_up
    """
    Ri_0 = maskExtractionCV(softmax <= 0, buffer)

    rundown_peaks = scipy.signal.find_peaks(
         Ri_0 - np.mean(Ri_0), prominence=np.std(Ri_0) / 2)[0]
    runup_peaks   = scipy.signal.find_peaks(
        -Ri_0 + np.mean(Ri_0), prominence=np.std(Ri_0) / 2)[0]

    xy = np.vstack([
        [0, 0.5],
        *sorted(
            [(t, 1) for t in runup_peaks] + [(t, 0) for t in rundown_peaks],
            key=lambda r: r[0]
        ),
        [softmax.shape[0], 0.5],
    ])

    Ri_down = maskExtractionCV(softmax <= rundown_val, buffer)
    Ri_up   = maskExtractionCV(softmax <= runup_val,   buffer)

    wt1 = scipy.interpolate.pchip_interpolate(xy[:, 0], xy[:, 1], np.arange(len(Ri_0)))
    Ri  = np.average([Ri_up, Ri_down], axis=0, weights=[wt1, 1 - wt1])
    return Ri, Ri_0, Ri_down, Ri_up


def getSoftMaxScores(metaDir, filename):
    """
    Load and concatenate softmax score arrays (.npy or .npz) matching filename.

    Inputs:
        metaDir  - directory containing softmax files (str or Path)
        filename - CIRN-formatted timestack filename
    Returns:
        av_prob_stack (np.ndarray)
    """
    metaDir  = Path(metaDir)
    # Strip two extensions: e.g. epoch.xxx.runup_timestack.tiff -> epoch.xxx
    fileBase = Path(Path(filename).stem).stem

    dateList = list(metaDir.glob(f'{fileBase}*.npz')) or \
               list(metaDir.glob(f'{fileBase}*.npy'))

    def _load(p):
        return np.load(p) if p.suffix == '.npy' else np.load(p)['av_prob_stack']

    return np.vstack([_load(p) for p in dateList])


def getSurveyVals(epochTime, site):
    """
    Return (xVals, yVals, zVals, dateObj_str) from the closest topo survey.
    Supports .mat (surveyInterp struct) and .xyz (CSV) formats via
    exampleTopoName in config.
    """
    dataDate = datetime.fromtimestamp(epochTime, tz=pytz.utc)

    if site.siteName == 'Dunex':
        from runup.extra_functions import loadDunexSurvey
        return loadDunexSurvey(site, dataDate)

    topoDir = Path(site.topoDir)
    topoFmt = getattr(site, 'exampleTopoName', None)

    if topoFmt:
        ext = Path(topoFmt).suffix
        if '<date>' in topoFmt:
            surveyFiles = list(topoDir.glob(topoFmt.replace('<date>', '*')))
            if not surveyFiles:
                raise FileNotFoundError(
                    f"No topo files matching '{topoFmt}' in {topoDir}")
            if len(surveyFiles) == 1:
                surveyPath = surveyFiles[0]
                dateObj    = datetime.fromtimestamp(
                                 surveyPath.stat().st_mtime).strftime('%b %d, %Y')
            else:
                surveyDate = searchSurveyDate(
                                 dataDate, [str(f) for f in surveyFiles], site=site)
                surveyPath = topoDir / topoFmt.replace('<date>', surveyDate)
                dateObj    = datetime.strptime(surveyDate, '%Y%m%d').strftime('%b %d, %Y')
        else:
            surveyPath = topoDir / topoFmt
            dateObj    = datetime.fromtimestamp(
                             surveyPath.stat().st_mtime).strftime('%b %d, %Y')
    else:
        # Legacy fallback for configs without exampleTopoName
        ext         = '.mat'
        topoPrefix  = 'topoSurveyInterpMaskedAllTopo'
        surveyFiles = list(topoDir.glob(f'{topoPrefix}*{site.topoSuffix}.mat'))
        if not surveyFiles:
            topoPrefix  = 'topoSurvey'
            surveyFiles = list(topoDir.glob(f'{topoPrefix}*{site.topoSuffix}.mat'))
        if len(surveyFiles) == 1:
            surveyPath = surveyFiles[0]
            dateObj    = datetime.fromtimestamp(
                             surveyPath.stat().st_mtime).strftime('%b %d, %Y')
        else:
            surveyDate = searchSurveyDate(
                             dataDate, [str(f) for f in surveyFiles], site=site)
            surveyPath = topoDir / f'topoSurveyInterpMaskedAllTopo{surveyDate}{site.topoSuffix}.mat'
            dateObj    = datetime.strptime(surveyDate, '%Y%m%d').strftime('%b %d, %Y')

    if ext == '.xyz':
        df = pd.read_csv(str(surveyPath), sep=',', header=None,
                         names=['x', 'y', 'z'])
        return df['x'].to_numpy(), df['y'].to_numpy(), df['z'].to_numpy(), dateObj

    survey = scipy.io.loadmat(str(surveyPath), squeeze_me=True, struct_as_record=False)
    sv     = survey['surveyInterp']
    return sv.x, sv.y, sv.z, dateObj


def getUVVals(site, epochTime=0, camNum='1', egFileString='pix_yyyymmdd'):
    """
    Return (U, V) pixel coordinate arrays for a site and epoch time.

    Supports structured (pix_YYYYMMDD/) and flat directory layouts.
    Always returns exactly (U, V) regardless of site.
    """
    if epochTime > 0:
        dataDate = datetime.fromtimestamp(epochTime, tz=pytz.utc)

    if site.siteName == 'Dunex':
        from runup.extra_functions import findDunexUVFile
        uvFileList = [str(f) for f in Path(site.uvTopDir).glob(
                          f'dunex_UV_c{camNum}_2021*.mat')]
        uvFile, _ = findDunexUVFile(uvFileList, dataDate)
        matStruct = scipy.io.loadmat(uvFile, squeeze_me=True, struct_as_record=False)
        return matStruct['UV'].U, matStruct['UV'].V

    # General case: try UV.txt first, then fall back to pix_* or flat .pix files
    try:
        if not hasattr(site, 'uvDir'):
            site.uvDir = site.uvTopDir
        uv = np.loadtxt(str(Path(site.uvDir) / 'UV.txt'), dtype=float).astype(int)
        return uv[:, 0], uv[:, 1]
    except Exception:
        pixelDirs = list(Path(site.uvTopDir).glob('pix_*'))
        if pixelDirs:
            pixelDateList = getPixDirDateList([str(p) for p in pixelDirs])
            ind           = getClosestPreviousDateIndex(pixelDateList, dataDate)
            site.uvDir    = pixelDirs[ind]
        else:
            site.uvDir = Path(site.uvTopDir)
        stem = site.saveName if site.saveName == 'madbeach' else site.siteName.lower()
        uv   = np.loadtxt(
                   str(Path(site.uvDir) / f'{stem}.c{camNum}.pix'),
                   dtype=float).astype(int)
        return uv[:, 0], uv[:, 1]


def findStackBreaks(U, V):
    """
    Identify stack break indices from pixel-distance discontinuities in UV.
    Returns: stackBreaks (n x 2 int array of [start, end] index pairs)
    """
    if len(V) == 0:
        return np.array([[0, 0]])
    if len(V) == 1:
        return np.array([[0, 1]])

    pix_dist    = np.concatenate(([0], np.sqrt(np.diff(U)**2 + np.diff(V)**2)))
    threshold   = max(np.mean(pix_dist) + np.std(pix_dist) * 2, 5)
    breakPoints = np.where(pix_dist > threshold)[0]

    if len(breakPoints) == 0:
        return np.array([[0, len(V)]])

    starts = np.concatenate(([0],         breakPoints))
    ends   = np.concatenate((breakPoints, [len(V)]))
    return np.column_stack((starts, ends))


# loadDunexTimestack, loadDunexSurvey moved to extra_functions.py


def loadMatStacks(filepath):
    """Load XYZ, T, RAW from a .mat timestack file."""
    mat = scipy.io.loadmat(str(filepath), squeeze_me=True, struct_as_record=False)
    return mat['XYZ'], mat['T'], mat['RAW']


def loadTiffStacks(site, filepath, camNum, sample_frequency_Hz=2):
    """
    Return (XYZ, T, RAW) from a .tiff timestack.

    T is constructed from the epoch time in the filename at the given
    sample frequency.
    """
    filepath  = Path(filepath)
    RAW       = plt.imread(str(filepath))
    epochTime = int(filepath.name.split('.')[0])
    endEpoch  = epochTime + RAW.shape[0] / sample_frequency_Hz
    T         = np.linspace(epochTime, endEpoch, num=RAW.shape[0]).astype('float')
    XYZ       = loadXYZ(site, camNum, epoch=epochTime)
    return XYZ, T, RAW


def loadXYZAndInstName(site, camNum, epoch=0):
    """
    Load local XYZ coordinates and instance type names for a site in one pass,
    avoiding opening the same .mat file twice.

    Layout is determined by config:
      - exampleRFileName without <date>: load that fixed filename from xyzTopDir
      - exampleRFileName with <date>: glob, date-match, load closest file
      - exampleRFileName not set: load <nName>.c<N>.r.mat from closest pix_*
        dir (or xyzTopDir directly for flat layouts)

    Returns:
        XYZ (np.ndarray, n x 3), InstName (list)
    """
    rFmt = getattr(site, 'exampleRFileName', None)

    if rFmt:
        if '<date>' in rFmt:
            fileDatetime = datetime.fromtimestamp(epoch) if epoch > 0 else datetime.now()
            matches      = sorted(Path(site.xyzTopDir).glob(rFmt.replace('<date>', '*')))
            if not matches:
                raise FileNotFoundError(
                    f"No r-files matching '{rFmt}' in {site.xyzTopDir}")
            surveyDate = searchSurveyDate(fileDatetime, [str(f) for f in matches])
            xyzPath    = Path(site.xyzTopDir) / rFmt.replace('<date>', surveyDate)
        else:
            xyzPath = Path(site.xyzTopDir) / rFmt
        mat = scipy.io.loadmat(str(xyzPath), squeeze_me=True, struct_as_record=False)
        return mat['XYZ'], []

    # Default: pix_* versioned or flat .r.mat layout
    stem = site.saveName if site.saveName == 'madbeach' else site.nName
    mat  = _load_pix_mat(site, 'xyzDir', epoch, lambda s: f'{stem}.c{camNum}.r.mat')
    return mat['r'].cams.XYZ, np.ndarray.tolist(mat['r'].cams.names)


def loadXYZ(site, camNum, epoch=0):
    """Load local XYZ coords. Returns n x 3 numpy array."""
    xyz, _ = loadXYZAndInstName(site, camNum, epoch)
    return xyz


def loadInstName(site, camNum, epoch=0):
    """Load instance type names (e.g. 'runup'). Returns list."""
    _, names = loadXYZAndInstName(site, camNum, epoch)
    return names


# local2latlon moved to extra_functions.py


def localTransformPoints(localOrigin, localAngle, directionFlag, Xin, Yin):
    """
    Transform between Local and Geographical World Coordinates.
    directionFlag: 1 = Geo→Local, 0 = Local→Geo
    """
    rad = np.deg2rad(localAngle)
    if directionFlag == 0:
        Xout = Xin * np.cos(rad) - Yin * np.sin(rad) + localOrigin[0]
        Yout = Yin * np.cos(rad) + Xin * np.sin(rad) + localOrigin[1]
    else:
        Ep   = Xin - localOrigin[0]
        Np   = Yin - localOrigin[1]
        Xout = Ep * np.cos(rad) + Np * np.sin(rad)
        Yout = Np * np.cos(rad) - Ep * np.sin(rad)
    return Xout, Yout


def maskExtractionCV(mask, buffer=[0, 0]):
    """Extract a runup contour from a binary mask using OpenCV connected components."""
    if any(buffer):
        mask_u8  = mask.astype(np.uint8)
        kernel   = np.ones((buffer[0], buffer[1]), np.uint8)
        dilated  = cv2.dilate(mask_u8, kernel, iterations=1)
        cnts, _  = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        filled   = cv2.fillPoly(np.zeros(mask.shape),
                                pts=[max(cnts, key=cv2.contourArea)], color=255)
        mask_bw  = np.multiply(mask_u8, filled) > 0
    else:
        mask_u8  = mask.astype(np.uint8)
        cnts, _  = cv2.findContours(mask_u8, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        mask_bw  = cv2.fillPoly(np.zeros(mask.shape),
                                pts=[max(cnts, key=cv2.contourArea)], color=255) > 0

    id_empty             = np.sum(mask_bw, 1) == 0
    mask_bw[id_empty, :] = mask_u8[id_empty, :]

    Ri      = first_greater_than(mask_bw, 1, 0).astype(float)
    nans, _ = nan_helper(Ri)
    x       = np.arange(mask.shape[0])
    valid   = ~nans & (Ri < mask_bw.shape[1]) & (Ri != 0)
    return np.interp(x, x[valid], Ri[valid])


def nan_helper(y):
    """Return (nan_mask, index_fn) for a 1D array with possible NaNs."""
    return np.isnan(y), lambda z: z.nonzero()[0]


# readDBFTopo moved to extra_functions.py


def readYAMLcomments(yamlfile):
    """Read comment lines from a YAML file into a dict keyed by first token."""
    comment_dict = {}
    with open(yamlfile, 'r') as f:
        for line in f:
            if line.startswith('#'):
                line = line.lstrip('#').strip()
                if line:
                    comment_dict[line.split()[0] + '_comment'] = '#' + line
    return comment_dict


# removeBlackBorders, searchUVFile, timestackFromJpg, timestackFromNc moved to extra_functions.py


def searchSurveyDate(dataDate, surveyFiles, site=None):
    """
    Find the survey file closest in date to dataDate.
    Returns: surveyDate string (yyyymmdd)
    """
    bestDate, bestDiff = None, None
    for filepath in surveyFiles:
        name = Path(filepath).name
        if site is not None and site.siteName == 'Dunex':
            if name[:4] == '2020':
                continue
            parts      = name.split('_')
            month, day = parts[-2][:2], parts[-2][2:4]
            dateObj    = datetime(2021, int(month), int(day)).replace(tzinfo=pytz.utc)
            surveyDate = '2021' + month + day
        else:
            surveyDate = name[-12:-4]
            dateObj    = datetime.strptime(surveyDate, '%Y%m%d').replace(tzinfo=pytz.utc)

        diff = abs((dataDate - dateObj).total_seconds())
        if bestDiff is None or diff < bestDiff:
            bestDiff = diff
            bestDate = surveyDate
    return bestDate


def timestackFromRawDirTop(site):
    """
    Populate site.timestackList from rawDirTop.
    Supports structured (rawDirTop/YYYY/cx/day/) and flat layouts.
    """
    site.timestackList = []
    rawDirTop = Path(site.rawDirTop)
    yearDirs  = list(rawDirTop.glob('20*'))

    if yearDirs:
        for yearPath in yearDirs:
            cxFolder = yearPath / 'cx'
            if not cxFolder.exists():
                continue
            for dayPath in cxFolder.iterdir():
                site.timestackList.extend(
                    str(f) for f in dayPath.glob(f'*runup{site.rNumber}.mat'))
    else:
        site.timestackList.extend(
            str(f) for f in rawDirTop.glob(f'*runup{site.rNumber}.mat'))


# trimXYZ moved to extra_functions.py


def uniqueYamlDates(yamlList):
    """
    Return a sorted list of unique calibration dates parsed from YAML filenames.
    Convention: <saveName>_c<N>_<YYYYMMDD>_<suffix>.yaml
    """
    dates = set()
    for f in yamlList:
        d = Path(f).name.split('_')[2]
        dates.add(datetime(int(d[:4]), int(d[4:6]), int(d[6:8])))
    return sorted(dates)


# unix2datetime moved to extra_functions.py


def yaml2dict(yamlfile):
    """Load a YAML file and return its contents as a dict."""
    with open(yamlfile, 'r') as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
            return None
