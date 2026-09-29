
## run after running 'batch_seg_images_in_folder.py'
# which creates a folder of 'predseg' outputs for mapping

# import numpy as np
# HPC/headless change: tkinter GUI imports removed.
# PREVIOUS: from tkinter import Tk
# PREVIOUS: from tkinter.filedialog import askopenfilename, askdirectory
import os 
from glob import glob

## prepare 4 text files. Each lists folders to process. One file for output (predseg) files, one for xmls, one for wlds, and finally one for output folders

# HPC/headless change. PREVIOUS: Tk askopenfilename dialog.
folderlist = input("Path to text file listing model output folders to process: ").strip().strip('"').strip("'")
print(folderlist)

with open(folderlist) as f:
    folders = f.readlines()
folders = [f.strip() for f in folders]
print("{} folders to process".format(len(folders)))

# HPC/headless change. PREVIOUS: Tk askopenfilename dialog.
folderlist = input("Path to text file listing wld folders to process: ").strip().strip('"').strip("'")
print(folderlist)

with open(folderlist) as f:
    wlds = f.readlines()
wlds = [f.strip() for f in wlds]
print("{} folders to process".format(len(wlds)))

# HPC/headless change. PREVIOUS: Tk askopenfilename dialog.
folderlist = input("Path to text file listing xml folders to process: ").strip().strip('"').strip("'")
print(folderlist)

with open(folderlist) as f:
    xmls = f.readlines()
xmls = [f.strip() for f in xmls]
print("{} folders to process".format(len(xmls)))

# HPC/headless change. PREVIOUS: Tk askopenfilename dialog.
folderlist = input("Path to text file listing output folders: ").strip().strip('"').strip("'")
print(folderlist)

with open(folderlist) as f:
    outputs = f.readlines()
outputs = [f.strip() for f in outputs]
print("{} folders to process".format(len(outputs)))


## cycle through each group of folders
for output, xml, wld, folder in zip(outputs, xmls, wlds, folders):

    files = glob(folder+os.sep+'*predseg.png')
    print("Found {} files".format(len(files)))
    print('Copying flles to '+output)
    os.system('cp '+xml+'/*.xml '+output)
    os.system('cp '+wld+'/*.wld '+output)
    os.system('cp '+folder+'/*predseg.png '+output)

    print('Converting png to jpg files')
    os.system('for f in '+output+'/*.png; do convert $f -quality 100 "${f%_predseg.png}.jpg"; done')

    print('Converting jpg to geotiff files')
    os.system('for k in '+output+'/*.jpg; do gdal_translate -of GTiff -b 1 $k "${k%jpg}tif"; done')

    print('Making large mosaic geotiff file')
    os.system('gdal_merge.py -o '+output+'/mosaic.tif -of GTiff -co "COMPRESS=JPEG" -co "TILED=YES"  '+output+'/*.tif')