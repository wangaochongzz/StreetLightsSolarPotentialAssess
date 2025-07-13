# StreetLightsSolarPotentialAssess
This library is used to store data and code for evaluating the solar potential of streetlights through Street View imagery and deep learning. The operating environment used to develop the code for this library is Windows 11 and Python 3.7.

### 1. The first folder streetlight_detection contains the sample streetlight header dataset, YOLOv5 model, and DepplabV3+ model.
Environment configuration 
``` pip install -r requirement ```
model training 
```python train.py ```
model validation 
```python get_map.py ```
lamp head detection 
```python predict.py```

### 2. The second folder streetlight_location contains the monodepth2 model and the triangulation model for locating streetlights.
Depth estimation 
```3_depth_prediction_dir.ipynb```
Multi-view location 
```python objectmapping.py```

### 3. The third folder, solar_assess, contains the code for evaluating solar radiation from streetlights using Street View images.
Generate a fisheye image 
```01_fisheye_images_generation.ipynb``` 
Fisheye image superimposed on the trajectory of the sun 
``` 02_overlay_the_trajectory_of_the_sun.ipynb ```
Calculate_the_radiation_value_received_by_the_PV_panel 
``` 03_calculate_solar_radiation .ipynb```
