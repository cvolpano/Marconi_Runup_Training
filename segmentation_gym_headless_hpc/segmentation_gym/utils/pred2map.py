
## run after running 'seg_images_in_folder.py'
# which creates a folder of 'predseg' outputs for mapping


import numpy as np
# HPC/headless change: tkinter GUI imports removed.
# PREVIOUS: from tkinter import Tk
# PREVIOUS: from tkinter.filedialog import askopenfilename, askdirectory
import os 
from glob import glob

# HPC/headless change. PREVIOUS: Tk askdirectory dialog.
folder = input("Path to directory of model output (label) files: ").strip().strip('"').strip("'")
print(folder)

files = glob(folder+os.sep+'*predseg.png')
print("Found {} files".format(len(files)))

# HPC/headless change. PREVIOUS: Tk askdirectory dialog.
wld = input("Path to directory of wld files: ").strip().strip('"').strip("'")
print(wld)

# HPC/headless change. PREVIOUS: Tk askdirectory dialog.
xml = input("Path to directory of xml files: ").strip().strip('"').strip("'")
print(xml)

# HPC/headless change. PREVIOUS: Tk askdirectory dialog.
output = input("Path to directory to store outputs: ").strip().strip('"').strip("'")
print(output)

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