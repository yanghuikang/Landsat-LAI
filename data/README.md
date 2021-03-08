### LAI training data
The zip file was split into four parts. Please download all the parts.

    LAI_train_sample_CONUS_v0.1.1_part.z01
    LAI_train_sample_CONUS_v0.1.1_part.z02
    LAI_train_sample_CONUS_v0.1.1_part.z03
    LAI_train_sample_CONUS_v0.1.1_part.zip

On mac or linux, first combine parts into a single zip file.  

    cat LAI_train_sample_CONUS_v0.1.1_part.* > LAI_train_sample_CONUS_v0.1.1_all.zip
   
Then unzip the single zip, will result in a 370MB csv file. 

    unzip LAI_train_sample_CONUS_v0.1.1_all.zip

The "UID" column encodes the geographic coordiantes (latitude and longitude) and the Landsat image information, in the following format "LATITUTE\_LONGITUDE\_SENSOR\_PATH/ROW\_DATE". For example, "N03979375000\_W08188778050\_LT05\_018032\_20080802" means the following,  

* Latitude: 39.79375 North
* Longitude: 81.8877805 West
* Landsat sendor: Landsat 5 TM
* WRS Path: 18
* WRS Row: 32
* Image date: 08/02/2008

These information can uniqely identify a Landsat surface reflectance image and the centroid corresponding to a MODIS LAI pixel.  

The coding for "biome2" is as follows.  

biome2 code | Biome type | NLCD land cover
------------|------------|----------------
1|Deciduous Forest|Deciduous Forest (41)
2|Evergreen Forest|Evergreen Forest (42)
3|Mixed Forest|Mixed Forest (43)
4|Shrub|Shrubland (52)
5|Grass|Grassland (71); Pasture (72)
6|Cropland|Cultivated Crops (82)
7|Woody Wetlands|Woody Wetlands (90)
8|Herbaceous Wetlands|Emergent Herbaceous Wetlands (95)


### Convex hull data
    LAI_train_convex_hull_CONUS_v0.1.1.csv
The convex hull information used to generate input out-of-range flag for the QA layer.