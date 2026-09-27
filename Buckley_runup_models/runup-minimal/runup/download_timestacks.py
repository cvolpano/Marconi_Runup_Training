import logging
import argparse
import pytz
import fsspec
from pathlib import Path
from datetime import datetime

from .pixel2runup import getSite


def download_timestacks(site_name):
    fileSystem = fsspec.filesystem('s3', profile='SPCoastCamKey-2024')
    configDir  = str(Path(__file__).resolve().parent.parent / 'config')
    site       = getSite(site_name, configDir)
    stackDir   = Path(site.stackDir)

    logDir = Path(__file__).resolve().parent.parent / 'logs'
    logDir.mkdir(exist_ok=True)
    logging.basicConfig(
        filename=str(logDir / f'{site.siteName}_missing_files.log'),
        format='%(asctime)s - %(message)s'
    )

    print(f'----- Downloading {site.siteName} files from AWS -----')

    if not site.s3TopDir:
        print('No s3TopDir set — skipping download.')
        return

    # Derive transect string from exampleStackName, e.g. 'runup100'
    example_parts  = site.exampleStackName.split('.')
    match_transect = example_parts[8]

    for yearDict in fileSystem.listdir(f's3://{site.s3TopDir}/'):
        yearFolder = yearDict['Key']
        year       = yearFolder.split('/')[-1]
        if not (site.startDate.year <= int(year) <= site.endDate.year):
            continue

        for dayDict in fileSystem.listdir(f's3://{yearFolder}/'):
            dayFolder = dayDict['Key']
            dayOfYear = dayFolder.split('/')[-1][:3]
            date      = datetime.strptime(
                            f'{year}-{dayOfYear}', '%Y-%j').replace(tzinfo=pytz.utc)
            if not (site.startDate <= date <= site.endDate):
                continue

            for imageDict in fileSystem.listdir(f's3://{dayFolder}/'):
                image    = imageDict['Key']
                filename = image.split('/')[-1]
                parts    = filename.split('.')

                if parts[-1] not in ('tiff', 'mat'):
                    continue

                # Match site name fragment
                if site.nName not in filename.lower():
                    logging.info(f'Site mismatch - File: {filename}')
                    continue

                # Match runup transect from exampleStackName
                if len(parts) >= 9 and parts[8] != match_transect:
                    logging.info(f'Transect mismatch - File: {filename}, Expected: {match_transect}')
                    continue

                epoch = int(parts[0])
                if list(stackDir.glob(f'{epoch}*{match_transect}*')):
                    logging.info(f'File already exists: {filename}')
                    continue

                print(f'File : {filename}')
                fileSystem.get(image, str(stackDir / filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download timestacks from S3')
    parser.add_argument('site_name', help='Site name matching config file (e.g. Molokai_c2_runup03)')
    download_timestacks(parser.parse_args().site_name)
