# StreetLightsSolarPotentialAssess

This library is used to store data and code for evaluating the solar potential of streetlights through Street View imagery and deep learning. The operating environment used to develop the code for this library is Windows 11, Python 3.7 and pytorch.

### 1. The streetlight_detection folder holds example data and models for detecting streetlight heads and extracting poles.

yolov5 light head detection 

```python predict.py ```

deeplabv3+ extracts light poles 

```deeplabv3plus.ipynb ```

### 2. The streetlight_location folder holds the code to locate the streetlight's single-view location and multi-view location.
   
Monocular depth estimation 

```3_depth_prediction_dir.ipynb ```

Multi-view location 

```python objectmapping.py```

### 4. The solar_assess folder holds the code to calculate the solar radiation of the street light.

Generate a fisheye image 

```01_fisheye_images_generation.ipynb ```

Fisheye image superimposed on the trajectory of the sun 

 ```02_overlay_the_trajectory_of_the_sun.ipynb``` 

Calculate the radiation received by the PV panel 

 ```03_calculate_solar_radiation .ipynb```
