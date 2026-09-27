import argparse
import pytz
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta

from .pixel2runup import getSite, getUVVals, loadXYZ, loadInstName, findStackBreaks


def isolate_timestacks(site_name):
    configDir = str(Path(__file__).resolve().parent.parent / 'config')
    site      = getSite(site_name, configDir)
    stackDir  = Path(site.stackDir)

    # Match combined ras files by site name fragment
    timestackFiles = [
        f for f in stackDir.glob(f'*{site.nName}*.ras{site.stackExtension}')
        if len(f.name.split('.')) == 10
    ]

    print(f'----- Splitting {site.siteName} files into separate timestacks -----')
    for cc, filepath in enumerate(timestackFiles):
        print(f'File {cc} of {len(timestackFiles)}: {filepath}')
        parts     = filepath.name.split('.')
        epochTime = int(parts[0])

        if filepath.stat().st_size < 1_000_000:
            print(f'Skipping {filepath} - file size less than 1,000 kb')
            continue

        camNum   = site.camNum if parts[7][-1].isalpha() else int(parts[7][-1])
        fileDate = datetime.fromtimestamp(epochTime).replace(tzinfo=pytz.utc)
        if (fileDate < site.startDate) or (fileDate > site.endDate + timedelta(days=1)):
            continue

        img          = skimage.io.imread(str(filepath))
        U, V         = getUVVals(site, epochTime, camNum=camNum)
        XYZ          = loadXYZ(site, camNum=camNum, epoch=epochTime)
        InstName     = loadInstName(site, camNum=camNum, epoch=epochTime)
        runupIEnd    = InstName.count('runup')
        stackBreaks  = findStackBreaks(U, V)

        timestacksList = []
        for start_idx, end_idx in stackBreaks:
            if img.ndim > 2:
                if start_idx >= runupIEnd:
                    continue
                timestack = img[:, start_idx:end_idx, :]
            else:
                timestack = img[:, start_idx:end_idx]

            if np.mean(np.diff(XYZ[:, 0]) < 0) > 0.5:
                U         = np.flip(U)
                V         = np.flip(V)
                timestack = np.flip(timestack, axis=1)
            timestacksList.append(timestack)

        for i, timestack in enumerate(timestacksList):
            newFilename = filepath.name.replace(parts[8], f'runup{i + 1:02d}')
            plt.imsave(str(stackDir / newFilename), timestack)

        filepath.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Split raw timestacks into individual runup timestacks')
    parser.add_argument('site_name', help='Site name matching config file (e.g. MadBeach2022_c1_runup90)')
    isolate_timestacks(parser.parse_args().site_name)
