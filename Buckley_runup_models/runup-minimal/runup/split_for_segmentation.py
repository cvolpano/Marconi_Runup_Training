import argparse
import pytz
import skimage.io
import skimage.color
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta

from .pixel2runup import getSite


def split_for_segmentation(site_name):
    configDir = str(Path(__file__).resolve().parent.parent / 'config')
    site      = getSite(site_name, configDir)
    site.gatherTimestacks()

    outputFolder = Path(site.segDir)
    outputFolder.mkdir(parents=True, exist_ok=True)

    # Exclude raw/rbr files not yet ready for segmentation
    timestackFiles = [
        f for f in site.timestackList
        if '.ras' not in Path(f).name and '.rbr' not in Path(f).name
    ]

    # Group by camera string parsed directly from filename (e.g. 'c1', 'c2', 'cx')
    camStrings = sorted({Path(f).name.split('.')[7] for f in timestackFiles})

    for camStr in camStrings:
        camFiles = [f for f in timestackFiles if camStr in Path(f).name]

        existingEpochs = {
            int(f.name.split('.')[0])
            for f in outputFolder.glob(f'*.{camStr}.*')
            if f.suffix in ('.jpg', '.npy')
        }

        doneNcEpochs = set()
        if not site.writeOverNC:
            doneNcEpochs = {
                int(f.name.split('.')[0])
                for f in Path(site.netcdfDir).glob(f'*.{camStr}.*.nc')
            }

        print(f'----- preparing images for ML model ({camStr}) -----')
        for cc, file in enumerate(camFiles):
            filepath = Path(file)
            parts    = filepath.name.split('.')
            print(f'File {cc} of {len(camFiles)}: {filepath}')

            epochTime = int(parts[0])

            if epochTime in existingEpochs:
                print(f'File already exists in output folder: {filepath.name}')
                continue
            if epochTime in doneNcEpochs:
                print(f'File already processed in nc folder: {filepath.name}')
                continue

            fileDate = datetime.fromtimestamp(epochTime).replace(tzinfo=pytz.utc)
            if (fileDate < site.startDate) or (fileDate > site.endDate + timedelta(days=1)):
                continue

            if site.stackExtension == '.mat':
                import scipy.io as sio
                img = sio.loadmat(str(filepath))['RAW']
            else:
                img = skimage.io.imread(str(filepath))

            if img.ndim == 3:
                img = skimage.color.rgb2gray(img[:, :, :3])

            # Split into ~1024-row sections for the ML model input size
            target_height = 1024
            num_sections  = max(1, int(np.round(img.shape[0] / target_height)))
            cuts          = np.round(
                                np.linspace(0, img.shape[0], num_sections + 1)
                            ).astype(int)

            for i, (start, end) in enumerate(zip(cuts[:-1], cuts[1:])):
                out_path = outputFolder / f'{filepath.stem}-{i:02d}.jpg'
                plt.imsave(str(out_path), img[start:end, :], cmap=plt.cm.gray)

        print(f'Saved split images to {outputFolder}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare timestacks for segmentation model')
    parser.add_argument('site_name', help='Site name matching config file (e.g. MadBeach2022_c1_runup90)')
    split_for_segmentation(parser.parse_args().site_name)
