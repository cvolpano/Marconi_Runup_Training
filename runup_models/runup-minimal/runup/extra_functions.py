# Functions not currently used by the main pipeline but kept for reference.

from pathlib import Path
import utm, calendar
import numpy as np
import numpy.matlib
import scipy
import pytz
import pandas
import requests
import json
import yaml
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from dbfread import DBF
from datetime import datetime, timezone, date, timedelta

from runup.pixel2runup import localTransformPoints, searchSurveyDate


def interpTopoCrossShoreNearest(xRaw, yRaw, xiRaw, yiRaw, xWalk, yWalk, zWalk, xlim, ylim):
    """
    Topo/bathy interpolation along a cross-shore transect.

    Rotates coordinates to align with the transect, bins available survey data
    and takes the mean, expanding the alongshore bin until data is found.
    """
    xRaw  = xRaw.flatten();  yRaw  = yRaw.flatten()
    xiRaw = xiRaw.flatten(); yiRaw = yiRaw.flatten()
    xWalk = xWalk.flatten(); yWalk = yWalk.flatten()
    zWalk = zWalk.flatten()

    transect_origin = xRaw[0], yRaw[0]
    transect_angle  = np.rad2deg(np.arctan2(yRaw[-1] - yRaw[0], xRaw[-1] - xRaw[0]))
    xRaw_rot, yRaw_rot = localTransformPoints(transect_origin, transect_angle, 1, xRaw, yRaw)

    if xiRaw.size != 0:
        xiRaw_rot, yiRaw_rot = localTransformPoints(
            transect_origin, transect_angle, 1, xiRaw, yiRaw)
        good     = np.argwhere(~np.isnan(xiRaw_rot))
        xRaw_rot = xiRaw_rot[good].flatten()
        yRaw_rot = yiRaw_rot[good].flatten()

    xWalk_rot, yWalk_rot = localTransformPoints(
        transect_origin, transect_angle, 1, xWalk, yWalk)
    xWalk_rot = xWalk_rot.flatten()
    yWalk_rot = yWalk_rot.flatten()

    id = abs(yWalk_rot) < ylim
    plt.plot(xWalk_rot[id], yWalk_rot[id], 'k.')
    plt.plot(xRaw_rot,      yRaw_rot,      'r.')
    plt.xlim(np.min(xRaw_rot) - xlim, np.max(xRaw_rot) + xlim)
    plt.ylim(-ylim, ylim)
    plt.show()

    dstep = 5
    dx    = 1
    x_interp_rot      = np.arange(
        np.min(xRaw_rot.round()) - xlim, np.max(xRaw_rot.round()) + xlim, dx)
    yWalk_rot_interp  = np.interp(xWalk_rot, xRaw_rot, yRaw_rot, right=np.nan, left=np.nan)

    bins              = np.append(x_interp_rot - dx/2, x_interp_rot[-1] + dx/2)
    zSurvey_rot_interp = x_interp_rot * np.nan
    zSurvey_rot_hist   = x_interp_rot * np.nan

    for i in np.arange(dstep, ylim, dstep):
        id    = abs(yWalk_rot - yWalk_rot_interp) <= i
        tmp_x = xWalk_rot[id]; tmp_z = zWalk[id]
        tmp_z = tmp_z[tmp_x.argsort()]; tmp_x = tmp_x[tmp_x.argsort()]
        tmp_z_interp = np.interp(x_interp_rot, tmp_x, tmp_z, right=np.nan, left=np.nan)
        zSurvey_rot_interp[np.isnan(zSurvey_rot_interp)] = \
            tmp_z_interp[np.isnan(zSurvey_rot_interp)]
        bin_means, _, _ = scipy.stats.binned_statistic(
            tmp_x, tmp_z, statistic='mean', bins=bins)
        zSurvey_rot_hist[np.isnan(zSurvey_rot_hist)] = \
            bin_means[np.isnan(zSurvey_rot_hist)]

    id = abs(yWalk_rot) <= 12
    plt.plot(xWalk_rot[id], zWalk[id],        'b.')
    plt.plot(x_interp_rot,  zSurvey_rot_hist, 'r.')
    plt.show()

    dx = 0.1; dy = 0.1
    xSurvey_rot, ySurvey_rot = np.meshgrid(
        np.arange(np.min(xRaw_rot.round()) - xlim, np.max(xRaw_rot.round()) + xlim, dx),
        np.arange(np.round(np.mean(yRaw_rot)) - ylim, np.round(np.mean(yRaw_rot)) + ylim, dy)
    )
    xSurvey, ySurvey = localTransformPoints(
        transect_origin, transect_angle, 0, xSurvey_rot, ySurvey_rot)

    good     = np.argwhere(~np.isnan(zSurvey_rot_hist))
    zSurvey  = np.interp(xSurvey_rot[0, :],
                         x_interp_rot[good].flatten(),
                         zSurvey_rot_hist[good].flatten(),
                         right=np.nan, left=np.nan)
    zSurvey  = np.matlib.repmat(zSurvey.T, xSurvey_rot.shape[0], 1)

    nan_mask = np.ones(xSurvey_rot[0, :].size)
    if any(np.isnan(zSurvey_rot_hist)):
        coeff = np.polyfit(x_interp_rot[good].flatten(),
                           zSurvey_rot_hist[good].flatten(), 1)
    if np.isnan(zSurvey_rot_hist[0]):
        nan_mask[0:np.min(good) - 1] = 0
        zSurvey_rot_hist[0] = ((x_interp_rot[np.min(good)] - x_interp_rot[0])
                                * -coeff[0] + zSurvey_rot_hist[np.min(good)])
    if np.isnan(zSurvey_rot_hist[-1]):
        nan_mask[np.max(good) + 1:-1] = 0
        zSurvey_rot_hist[-1] = ((x_interp_rot[-1] - x_interp_rot[np.max(good)])
                                 * coeff[0] + zSurvey_rot_hist[np.max(good)])

    good           = np.argwhere(~np.isnan(zSurvey_rot_hist))
    zSurvey_Extend = np.interp(xSurvey_rot[0, :],
                               x_interp_rot[good].flatten(),
                               zSurvey_rot_hist[good].flatten())
    zSurvey_Extend = np.matlib.repmat(zSurvey_Extend.T, xSurvey_rot.shape[0], 1)

    return xSurvey, ySurvey, zSurvey, zSurvey_Extend, nan_mask


def createNcFilename(oldFilename, site, rNumberCx):
    """
    Convert a short 4-element filename to a long CIRN-formatted 10-element filename.
    """
    epochTime   = oldFilename.split('.')[0]
    _, datetimeObj = unix2datetime(epochTime)

    year  = str(datetimeObj.year)
    month = str(datetimeObj.month)
    day   = str(datetimeObj.day).zfill(2)
    hour  = str(datetimeObj.hour).zfill(2)
    minute = str(datetimeObj.minute).zfill(2)
    second = str(datetimeObj.second).zfill(2)

    dayOfWeek      = calendar.day_name[datetimeObj.weekday()][:3]
    monthFormatted = calendar.month_name[int(month)][:3]

    return (f"{epochTime}.{dayOfWeek}.{monthFormatted}.{day}_{hour}_{minute}_{second}"
            f".GMT.{year}.{site}.cx.runup{rNumberCx}.runup_timestack.nc")


def unix2datetime(unixnumber):
    """
    Get a datetime object and UTC string from a unix/epoch time string.
    The last digit is zeroed to align to 10-second boundaries (legacy behavior).
    """
    try:
        timeStamp   = int(unixnumber[:-1] + '0')
        datetimeObj = datetime.fromtimestamp(timeStamp, tz=timezone.utc)
        datetimeStr = datetimeObj.strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        print(e)
        raise
    return datetimeStr, datetimeObj


def datumFromDatetime(fileDatetime, tideData):
    """Extract a single water level datum from a NOAA tide gauge Pandas dataframe."""
    fileTimestamp = pandas.Timestamp(
        fileDatetime.year, fileDatetime.month, fileDatetime.day,
        fileDatetime.hour, fileDatetime.minute)
    row = tideData[tideData['Date Time'] == fileTimestamp]
    return row['Water Level'].values[0]


def getPressure(beginDate, endDate, fileDatetime, spotterId, spotterToken):
    """Query Sofar spotter buoy API for pressure data."""
    beginDateStr = beginDate.strftime('%Y-%m-%dT%H') + ':00:00Z'
    endDateStr   = endDate.strftime('%Y-%m-%dT%H')   + ':00:00Z'

    url = (f'https://api.sofarocean.com/api/sensor-data'
           f'?spotterId={spotterId}&startDate={beginDateStr}&endDate={endDateStr}')
    smartMooringTable = json.loads(
        requests.get(url, headers={'token': spotterToken}).content)['data']
    meanPressureTS, meanPressureInst, meanPressureDatetimes = \
        extractMeanPressure(smartMooringTable, fileDatetime)

    url = (f'https://api.sofarocean.com/api/wave-data'
           f'?spotterId={spotterId}&startDate={beginDateStr}&endDate={endDateStr}'
           f'&includeBarometerData=true')
    wavesDataTable = json.loads(
        requests.get(url, headers={'token': spotterToken}).content)['data']
    barometricPressureTS, barometricPressureInst, barometricPressureDatetimes = \
        extractBarometricPressure(wavesDataTable['barometerData'], fileDatetime)

    pressureInst = meanPressureInst - barometricPressureInst
    pressureTS   = subtractPressureTS(
        meanPressureTS, barometricPressureTS,
        meanPressureDatetimes, barometricPressureDatetimes)
    return pressureTS, pressureInst


def extractBarometricPressure(dataTable, fileDatetime):
    """Extract barometric pressure timeseries from Sofar spotter waves data."""
    pressureTS, pressureDatetimes = [], []
    fileDatetime = fileDatetime.replace(tzinfo=None)
    closestIndex = None
    lowestTimeDiff = None

    for i, row in enumerate(dataTable):
        if row['unit_type'] != 'barometric_pressure':
            continue
        pressureTS.append(row['value'])
        pressureDatetimes.append(row['timestamp'])
        dt   = datetime.strptime(row['timestamp'], '%Y-%m-%dT%H:%M:%S.%fZ')
        diff = abs(fileDatetime - dt)
        if lowestTimeDiff is None or diff < lowestTimeDiff:
            lowestTimeDiff = diff
            closestIndex   = i

    return np.array(pressureTS), dataTable[closestIndex]['value'], pressureDatetimes


def extractMeanPressure(dataTable, fileDatetime):
    """Extract mean pressure timeseries from Sofar spotter smart mooring data."""
    pressureTS, pressureDatetimes = [], []
    fileDatetime = fileDatetime.replace(tzinfo=None)
    closestIndex = None
    lowestTimeDiff = None

    for i, row in enumerate(dataTable):
        if row['units'] == '°C':
            continue
        if row['data_type_name'] == 'rbrcoda3_meanpressure_21bits':
            pressureTS.append(row['value'])
            pressureDatetimes.append(row['timestamp'])
            dt   = datetime.strptime(row['timestamp'], '%Y-%m-%dT%H:%M:%S.%fZ')
            diff = abs(fileDatetime - dt)
            if lowestTimeDiff is None or diff < lowestTimeDiff:
                lowestTimeDiff = diff
                closestIndex   = i

    return np.array(pressureTS), dataTable[closestIndex]['value'], pressureDatetimes


def subtractPressureTS(meanPressureTS, barometricPressureTS,
                       meanPressureDatetimes, barometricPressureDatetimes):
    """Subtract barometric pressure timeseries from mean pressure timeseries."""
    meanDates  = [d[:13] for d in meanPressureDatetimes]
    baroDates  = [d[:13] for d in barometricPressureDatetimes]
    return np.array([meanPressureTS[meanDates.index(baroDates[i])] - barometricPressureTS[i]
                     for i in range(len(baroDates))])


def findDunexUVFile(uvList, dataDate):
    """Given a datetime, find the Dunex UV file whose date range contains it."""
    for file in uvList:
        parts        = Path(file).name.split('_')
        startDateStr = parts[4]
        endDateStr   = parts[5]
        startDate    = datetime(2021, int(startDateStr[:2]), int(startDateStr[2:4]),
                                hour=0).replace(tzinfo=pytz.utc)
        endDate      = datetime(2021, int(endDateStr[:2]),   int(endDateStr[2:4]),
                                hour=23).replace(tzinfo=pytz.utc)
        if startDate <= dataDate <= endDate:
            return file, startDate


def findRightBound(RAW):
    """Find the right bound of valid (non-zero) data in the runup timestack."""
    zeroTest       = np.array([0, 0, 0])
    flippedRAW     = np.flip(RAW, axis=1)
    for i in range(flippedRAW.shape[0]):
        for j in range(flippedRAW.shape[1]):
            if not np.array_equal(flippedRAW[i, j], zeroTest):
                return flippedRAW.shape[1] - j
    return RAW.shape[1]


def findLeftBound(RAW):
    """Find the left bound of valid (non-zero) data in the runup timestack."""
    zeroTest = np.array([0, 0, 0])
    for i in range(RAW.shape[0]):
        for j in range(RAW.shape[1]):
            if not np.array_equal(RAW[i, j], zeroTest):
                return j
    return 0


def removeBlackBorders(RAW, XYZ, returnBounds=False):
    """Remove black borders on the left and right sides of the raw timestack."""
    leftBound  = findLeftBound(RAW)
    rightBound = findRightBound(RAW)
    newRAW = RAW[:, leftBound:rightBound, :]
    newXYZ = XYZ[leftBound:rightBound, :]
    if returnBounds:
        return newRAW, newXYZ, leftBound, rightBound
    return newRAW, newXYZ


def getCamEastingNorthing(camGeom):
    """Return camera Easting and Northing UTM coordinates from a camera struct."""
    camInfoCamera   = camGeom.camera
    camInfoSite     = camGeom.site
    siteUTM         = utm.from_latlon(camInfoSite.lat, camInfoSite.lon)
    siteLocalOrigin = np.array([siteUTM[0], siteUTM[1]])
    return localTransformPoints(
        siteLocalOrigin, camInfoSite.degFromN, 0, camInfoCamera.x, camInfoCamera.y)


def local2latlon(localOrigin, localAngle, Xin, Yin, zoneNum, zoneLetter):
    """Convert local coordinates to latitude and longitude."""
    Easting, Northing = localTransformPoints(localOrigin, localAngle, 0, Xin, Yin)
    return utm.to_latlon(Easting, Northing, zoneNum, zoneLetter)


def readDBFTopo(dbfFile):
    """Read topo survey xyz values from a .dbf file."""
    surveyTable = DBF(dbfFile, load=True)
    xSurvey, ySurvey, zSurvey = [], [], []
    for rec in surveyTable.records:
        xSurvey.append(rec['Easting'])
        ySurvey.append(rec['Northing'])
        zSurvey.append(rec['Elevation'])
    return np.array(xSurvey), np.array(ySurvey), np.array(zSurvey)


def searchUVFile(dataDate, uvFileList):
    """Find the UV file closest in date to dataDate."""
    bestFile, bestDiff = None, None
    for filepath in uvFileList:
        d       = Path(filepath).name.split('_')[1].replace('.mat', '')
        dateObj = datetime.strptime(d, '%Y%m%d').replace(tzinfo=pytz.utc)
        diff    = abs((dataDate - dateObj).total_seconds())
        if bestDiff is None or diff < bestDiff:
            bestDiff = diff
            bestFile = Path(filepath).name
    return bestFile


def timestackFromJpg(jpgFile, rawDirTop, fileExtension):
    """Given a jpg file, return the path to the corresponding raw timestack file."""
    filename      = Path(jpgFile).name.replace('.runup_timestack.jpg', fileExtension)
    epochTime     = int(filename.split('.')[0])
    datetimeGMT   = datetime.fromtimestamp(epochTime, tz=pytz.utc)
    formattedDay  = datetimeGMT.strftime('%j_%b.%d')
    return str(Path(rawDirTop) / str(datetimeGMT.year) / 'cx' / formattedDay / filename)


def timestackFromNc(ncFile, rawDirTop, fileExtension):
    """Given a netCDF file, return the path to the corresponding raw timestack file."""
    filename     = Path(ncFile).stem + fileExtension
    epochTime    = int(filename.split('.')[0])
    datetimeGMT  = datetime.fromtimestamp(epochTime, tz=pytz.utc)
    formattedDay = datetimeGMT.strftime('%j_%b.%d')
    return str(Path(rawDirTop) / str(datetimeGMT.year) / 'cx' / formattedDay / filename)


def loadDunexTimestack(timestack):
    """Load XYZ, T, and RAW from a Dunex timestack .mat file."""
    runup    = timestack['runup']
    XYZ      = np.ndarray(shape=(runup.X.size, 3))
    XYZ[:, 0] = runup.X
    XYZ[:, 1] = runup.Y
    XYZ[:, 2] = runup.Z
    return XYZ, runup.T, np.rot90(runup.Irgb)


def loadDunexSurvey(site, dataDate):
    """Load Dunex XYZ survey values, selecting the closest survey to dataDate."""
    topoDir    = Path(site.topoDir)
    surveyFiles = [str(f) for f in topoDir.glob('*.xyz')]
    surveyDate = searchSurveyDate(dataDate, surveyFiles, site=site)
    dateObj    = datetime.strptime(surveyDate, '%Y%m%d').strftime('%b %d, %Y')
    matches    = list(topoDir.glob(
        f'2021029FA_PINWR_DUNEX_Topobathy_1m_{surveyDate[4:8]}*.xyz'))
    df = pandas.read_csv(str(matches[0]), sep=',', header=None,
                         names=['easting', 'northing', 'elevation'])
    return df['easting'].to_numpy(), df['northing'].to_numpy(), df['elevation'].to_numpy(), dateObj


def trimXYZ(site, filename, XYZ):
    """Trim XYZ array to match RAW dimensions for Dorado and Isla Verde."""
    if site.nName == 'dorado':
        rasNum = filename.split('.')[8][5]
        slices = {
            '0': (0, 568),   '1': (568, 1146),  '2': (1146, 1689), '3': (1689, 2245),
            '4': (0, 437),   '5': (437, 1128),   '6': (1128, 1876), '7': (1876, 2637),
        }
        s = slices.get(rasNum)
        if s:
            return XYZ[s[0]:s[1], :]
    elif site.nName == 'islaverde':
        return XYZ[0:1294, :]
    return XYZ
