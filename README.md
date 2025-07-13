# StreetLightsSolarPotentialAssess

This library is used to store data and code for evaluating the solar potential of streetlights through Street View imagery and deep learning. The operating environment used to develop the code for this library is Windows 11, Pytorch 3.10 and Python 3.7.

### 1. The streetlight_detection folder holds example data and models for detecting streetlight heads and extracting poles.

yolov5 streetlight head detection 

```python predict.py ```

deeplabv3+ extracts streetlight poles 

```deeplabv3plus.ipynb ```

### 2. The streetlight_location folder holds the code to locate the streetlight's single-view location and multi-view location.
   
streetlight monocular depth estimation 

```3_depth_prediction_dir.ipynb ```

streetlight multi-view location 

```python objectmapping.py```

### 3. The solar_assess folder holds the code that calculates the amount of solar radiation received by the streetlight.

generate fisheye image 

```01_fisheye_images_generation.ipynb ```

overlay the sun's trajectory

 ```02_overlay_the_trajectory_of_the_sun.ipynb``` 

calculate the radiation received by the solar streetlight 

 ```03_calculate_solar_radiation.ipynb```
