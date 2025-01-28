# Standalone scripts for the LAI algorithm

## Javascript
You can find code and examples in this GEE repo:   
https://code.earthengine.google.com/?accept_repo=users/kangyanghui/LAI

A lite example using the LAI library:   
[https://code.earthengine.google.com/275f132f46aad52dbf9ef15f3b7c11d2](https://code.earthengine.google.com/c4b56c53ff35d2b2bf8f385aa023fa5f)  
(or use the script path: https://code.earthengine.google.com/?scriptPath=users%2Fkangyanghui%2FLAI%3ALandsat_LAI_example)

A full standalone version:  
https://code.earthengine.google.com/cbbd5ad7ac5e5ee2815cd0137c8d2585  
(or use the script path: https://code.earthengine.google.com/?scriptPath=users%2Fkangyanghui%2FLAI%3ALandsat_LAI_example_full)

You may also download the standalone version: "Landsat\_LAI\_example\_full.js" within this folder.

## Python 
## The Python script is pending update. Please use the Javascript code.
This python3 script allows batch export of LAI images from Landsat 5/7/8/9 surface reflectance images using the [Google Earth Engine python client library](https://developers.google.com/earth-engine/guides/python_install). Before running the script, please make sure that the Earth Engine Library is installed. To run the script, specify the WRS path and row for a Landsat scene, the starting and end date, and an Earth Engine asset directory to write to. Before using this script, make sure that the [Google Earth Engine library](https://developers.google.com/earth-engine/guides/python_install) is installed.

### Usage
    python ee_Landsat_LAI_export.py -o <asset_dir> -p <path> -r <row> 
        -s <start_date> -d <end_date>
        
    Required arguments:
    -o  Earth Engine asset directory to export LAI images, can a foler or 
        an image collection
    -p  WRS Path number of the Landsat Collection 1 surface reflectance scene
    -r  WRS Row number of the Landsat Collection 1 surface reflectance scene
    -s  The start date to export image in YYYY-MM-dd
    -e  The end date (exclusive) to export image in YYYY-MM-dd

    Optional arguments:
    -h  show this help
    -v  Boolean to indicate whether LAI is computed for non-vegetative (e.g. urban)
        pixels (based on NLCD). 1 - generate; 0 - do not generate. Default is 0.

    Output:
    LAI maps exported as Google Earth Engine assets
    LAI images are automatically named following "LAI_<sensor>_<path/row>_<date>"
        e.g. LAI_LC08_044033_20180418

    Bands:
        LAI - LAI map scaled by 100 (scale factor = 0.01)
        QA - LAI quality band    
            QA is coded in a byte-size band using the least significant 3 bits
              Bit 0 - Input
                  0: Input within range
                  1: Input out-of-range
              Bit 1 - Output (LAI)
                  0: LAI within range (0-8)
                  1: LAI out-of-range
              Bit 2 - Biome
                  0: Vegetation (from NLCD scheme)
                  1: Non-vegetation (from NLCD scheme)
       
### Example
For example, 

    python ee_Landsat_LAI_export_v0.1.1.py -o projects/ee-yanghuikang/assets/LAI_test/LAI_test_v0_1_1 -p 44 -r 33 -s 2020-06-01 -e 2020-06-15
The output will look like, 

    assetDir: projects/ee-yanghuikang/assets/LAI_test/LAI_test_v0_1_1
    WRS path: 44
    WRS row: 33
    start date: 2020-06-01
    end date: 2020-06-15
    Number of Landsat images:  2
    CFBCSQCUGNZMKSDXF3HPE7EO LAI_LC08_044033_20200606
    X6JDL76R5BVHTW7JAGHFEJOA LAI_LE07_044033_20200614

This will export two LAI images to the designated EE asset image collection. The last two lines print the Task ID and the image name.
