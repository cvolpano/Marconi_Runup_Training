Doodleverse/Segmentation Gym SegFormer models for 2-class (water, other) segmentation of greyscale CoastCam runup timestack imagery

These SegFormer model data are based on greyscale timestack images of wave runup on beaches, and associated labels. The models have been used in support of Buckley et al. (in press). The models have been created using Segmentation Gym (Buscombe and Goldstein, 2022) using an as published dataset of images and associated label images. Image and auxiliary data come from:

1. Harrison, S.R., Buckley, M.L., Logan, J., Pomeroy, A.W.M., Storlazzi, C.D., Birchler, J.J., Palmsten, M.L., Swanson, E., and Johnson, E.L., 2024, USGS CoastCam at Waiakāne, Moloka'i, Hawai'i: 2018 timestack imagery and coordinate data: U.S Geological Survey data release, https://doi.org/10.5066/P14QO72F.

2. Harrison, S.R., Buckley, M.L., Johnson, C., Nowacki, D.J., Canals, M., Evans, C., Logan, J., Storlazzi, C.D., Birchler, J.J., Palmsten, M.L., Swanson, E., and Johnson, E.L., 2024, USGS CoastCam at Tres Palmas, Rincón, Puerto Rico: timestack imagery and coordinate data: U.S Geological Survey data release, https://doi.org/10.5066/P13QYBXP.

3. Harrison, S.R., Buckley, M.L., López Mújica, P., Logan, J., Cheriton, O., Viehman, T.S., Storlazzi, C.D., Birchler, J.J., Palmsten, M.L., Swanson, E., and Johnson, E.L., 2024, USGS CoastCam at Isla Verde, Puerto Rico: 2018-2019 timestack imagery and coordinate data: U.S Geological Survey data release, https://doi.org/10.5066/P13MJ66F.

4. Brown, J.A., Birchler, J.J., Palmsten, M.L., Swanson, E., Johnson, E.L., and Buckley, M.., 2024, USGS CoastCam at Sand Key, Florida: timestack imagery and coordinate data: U.S. Geological Survey data release, https://doi.org/10.5066/P13XR9TY.

5. Brown, J.A., Palmsten, M.L., Swanson, E., and Buckley, M., 2024, USGS CoastCam at Madeira Beach, Florida: timestack imagery and coordinate data: U.S. Geological Survey data release, https://doi.org/10.5066/P93VZION.



File descriptions

There are two models, SegFormer_Madeira_Duck_equal_finetune_Waiakane and SegFormer_Madeira_Duck_equal. These models are described in Buckley et al. (in review)

1. '.json' config file: this is the file that was used by Segmentation Gym* to create the weights file. It contains instructions for how to make the model and the data it used, as well as instructions for how to use the model for prediction. It is a handy wee thing and mastering it means mastering the entire Doodleverse.

2. '.h5' weights file: this is the file that was created by the Segmentation Gym* function `train_model.py`. It contains the trained model's parameter weights. It can called by the Segmentation Gym* function  `seg_images_in_folder.py`. Models may be ensembled.

3. '_model_history.npz' model training history file: this numpy archive file contains numpy arrays describing the training and validation losses and metrics. It is created by the Segmentation Gym function `train_model.py`

4. '.png' model training loss plot: this png file contains plots of training and validation scores during model training. A subset of data inside the .npz file. It is created by the Segmentation Gym function `train_model.py`

Classes: {0=other, 1=water}. Earlier versions of these models are available as Buscombe (2023)


Other references

Buscombe, D., & Goldstein, E. B. (2022). A reproducible and reusable pipeline for segmentation of geoscientific imagery. Earth and Space Science, 9, e2022EA002332. https://doi.org/10.1029/2022EA002332 See: https://github.com/Doodleverse/segmentation_gym
Buckley, M., Buscombe, D., Birchler, J.J., Palmsten, M.L., Swanson, E., Brown, J.A., Itzkin, M., Storlazzi, C.D., Harrison, S.R. (in review) Wave runup and total water level observations from time series imagery at several sites with varying nearshore morphologies. Coastal Engineering, in review
Buscombe, D. (2023). Doodleverse/Segmentation Gym Res-UNet models for 2-class (water, other) segmentation of CoastCam runup timestack imagery (v1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7921971

