#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate LAI maps Landsat images within the Contiguous US (CONUS)

LAI maps for all Landsat 5, 6, 8 surface reflectance images within a specified time
period for a specific path/row will be exported to an Earth Engine asset

NOTE: The algorithm should be applied to Landsat Collection 1 surface reflectance
images, currently only within CONUS. Work for Landsat 5, 7, and 8.
   
Requirements:
    earthengine-api>=0.1.232
    python>=3.6

Usage:
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
       

@author: Yanghui Kang (kangyanghui@gmail.com)
@License: MIT License

"""
import sys, getopt

try:
    import ee
except ModuleNotFoundError: 
    print('Please install earthengine-api before running this script.')
    
try:
    ee.Initialize()
except ee.EEException:
    ee.Authenticate()
    ee.Initialize()

LAI_version = '0.1.1'

def getAffineTransform(image):
    projection = image.projection()
    json = ee.Dictionary(ee.Algorithms.Describe(projection))
    return ee.List(json.get('transform'))

def renameLandsat(image):
    """
    Function that renames Landsat bands
    From landsat.
    """
    sensor = ee.String(image.get('SATELLITE'))
    from_list = ee.Algorithms.If(
        sensor.compareTo('LANDSAT_8'),
        ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'pixel_qa'],
        ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'pixel_qa'])
    to_list = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'pixel_qa']

    return image.select(from_list, to_list)


def getQABits(image, start, end, newName):
    """
     Function that returns an image containing just the specified QA bits.
    """
    # Compute the bits we need to extract.
    pattern = 0
    for i in range(start, end + 1):
       pattern = pattern + 2**i

    # Return a single band image of the extracted QA bits, giving the band
    # a new name.
    return image.select([0],[newName]).bitwiseAnd(pattern).rightShift(start)


def maskLST(image):
    """
    Function that masks a Landsat image based on the QA band
    """
    pixelQA = image.select('pixel_qa')
    cloud = getQABits(pixelQA, 1, 1, 'clear')
    return image.updateMask(cloud.eq(1))

def getLAIQA(landsat, sensor, lai):
    """
      QA is coded in a byte-size band occupying the least significant 3 bits
      Bit 0: Input
          0: Input within range
          1: Input out-of-range
      Bit 1: Output (LAI)
          0: LAI within range (0-8)
          1: LAI out-of-range
      Bit 2: Biome
          0: Vegetation (from NLCD scheme)
          1: Non-vegetation (from NLCD scheme)
      
      args: landsat - Landsat image (with 'biome2' band)
            sensor - "LT05"/"LE07"/"LC08"
            lai - computed lai image
    """

    # maximum for surface reflectance; minimum is always 0
    red_max = 5100
    green_max = 5100
    nir_max = 7100
    swir1_max = 7100
    lai_max = 8
  
    # information from the Landsat image
    # crs = landsat.select('red').projection().crs()
    # transform = getAffineTransform(landsat.select('red'))

    # Get pre-coded convex hull
    data = ee.FeatureCollection('projects/ee-yanghuikang/assets/LAI/LAI_train_convex_hull_by_sensor_v0_1_1')

    subset = data.filterMetadata('sensor','equals',sensor)
    subset = subset.sort('index')
    hull_array = subset.aggregate_array('in_hull')
    hull_array_reshape = ee.Array(hull_array).reshape([10,10,10,10])

    # rescale landsat image
    image_scaled = landsat.select('red').divide(red_max).multiply(10).floor().toInt() \
        .addBands(landsat.select('green').divide(green_max).multiply(10).floor().toInt()) \
        .addBands(landsat.select('nir').divide(nir_max).multiply(10).floor().toInt()) \
        .addBands(landsat.select('swir1').divide(swir1_max).multiply(10).floor().toInt())

    # get an out-of-range mask
    range_mask = landsat.select('red').gte(0) \
        .And(landsat.select('red').lt(red_max)) \
        .And(landsat.select('green').gte(0)) \
        .And(landsat.select('green').lt(green_max)) \
        .And(landsat.select('nir').gte(0)) \
        .And(landsat.select('nir').lt(nir_max)) \
        .And(landsat.select('swir1').gte(0)) \
        .And(landsat.select('swir1').lt(swir1_max))

    # apply convel hull and get QA Band
    hull_image = image_scaled.select('red').multiply(0).add(ee.Image(hull_array_reshape)) \
        .updateMask(range_mask)

    in_mask = hull_image \
        .arrayGet(image_scaled.select(['red','green','nir','swir1']).updateMask(range_mask))

    in_mask = in_mask.unmask(0).updateMask(landsat.select('red').mask()).Not().int()

    # check output range
    out_mask = lai.gte(0).And(lai.lte(lai_max)).updateMask(landsat.select('red').mask()).Not().int()

    # indicate non-vegetation biome
    biome_mask = landsat.select('biome2').eq(0).int()

    # combine
    qa_band = in_mask.bitwiseOr(out_mask.leftShift(1)).bitwiseOr(biome_mask.leftShift(2)).toByte()

    return qa_band.rename('QA')


def setDate(image):
    """
    Function that adds a "date" property to an image in format "YYYYmmdd"
    """

    eeDate = ee.Date(image.get('system:time_start'))
    date = eeDate.format('YYYYMMdd')
    return image.set('date',date)


def getRFModel(sensor, biome):
    """
    Wrapper function to train RF model given biome and sensor
    Args:
        sensor: str {'LT05', 'LE07', 'LC08'} (cannot be an EE object)
        biome: int

    from 'landsat.py'
    """

    filename = 'projects/ee-yanghuikang/assets/LAI/LAI_train_CONUS_v0_1_1'

    training_coll = ee.FeatureCollection(filename) \
        .filterMetadata('sensor', 'equals', sensor)

    # Get train sample by biome
    if biome > 0:
        training_coll = training_coll.filterMetadata('biome2', 'equals', biome)

    features = ['red', 'green', 'nir', 'swir1', 'lat', 'lon','NDVI', 'NDWI', 'sun_zenith', 'sun_azimuth']

    rf = ee.Classifier.smileRandomForest(numberOfTrees=100,minLeafPopulation=50,variablesPerSplit=5) \
        .setOutputMode('REGRESSION') \
        .train(features=training_coll,classProperty='MCD_LAI',inputProperties=features)

    return rf


def getTrainImg(image):
    """
    Takes an Landsat image and prepare feature bands
    """

    # Get NLCD for corresponding year
    nlcd_dict = {
        '2001': ['1997', '1998', '1999', '2000', '2001', '2002'],
        '2004': ['2003', '2004', '2005'],
        '2006': ['2006', '2007'],
        '2008': ['2008', '2009'],
        '2011': ['2010', '2011', '2012'],
        '2013': ['2013', '2014'],
        '2016': ['2015', '2016', '2017', '2018', '2019', '2020'],
    }
    nlcd_dict = ee.Dictionary({
        src_year: tgt_year
        for tgt_year, src_years in nlcd_dict.items()
        for src_year in src_years})
    nlcd_year = nlcd_dict.get(
        ee.Date(image.get('system:time_start')).get('year').format('%d'))
    nlcd_img = ee.ImageCollection('USGS/NLCD') \
        .filter(ee.Filter.eq('system:index', ee.String('NLCD').cat(nlcd_year))) \
        .first()

    # Add the NLCD year as a property to track which year was used
    image = image.set({'nlcd_year': nlcd_year})

    # Apply fmask
    image = maskLST(image)

    # Add the vegetation indices as additional bands
    NDVI = image.expression(
        'float((b("nir") - b("red"))) / (b("nir") + b("red"))')
    NDWI = image.expression(
        'float((b("nir") - b("swir1"))) / (b("nir") + b("swir1"))')

    image = image.addBands(NDVI.select([0], ['NDVI'])) \
                 .addBands(NDWI.select([0], ['NDWI']))


    # Map NLCD codes to biomes
    # CM - Added NLCD codes 11 and 12
    # CM - Switched from lists to a dictionary to improve readability
    nlcd_biom_remap = {
        11: 0, 12: 0,
        21: 0, 22: 0, 23: 0, 24: 0, 31: 0,
        41: 1, 42: 2, 43: 3, 52: 4,
        71: 5, 81: 5, 82: 6, 90: 7, 95: 8,
    }
    # fromList = [21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95]
    # toList = [0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 5, 6, 7, 8]
    biom_img = nlcd_img.remap(*zip(*nlcd_biom_remap.items()) )
    # biom_img = nlcd_img.remap(
    #     list(nlcd_biom_remap.keys()), list(nlcd_biom_remap.values()))

    # Add other bands

    # CM - Map all bands to mask image to avoid clip or updateMask calls
    mask_img = image.select(['pixel_qa'], ['mask']).multiply(0)
    image = image.addBands(mask_img.add(biom_img).rename(['biome2'])) \
        .addBands(mask_img.add(ee.Image.pixelLonLat().select(['longitude']))
                    .rename(['lon'])) \
        .addBands(mask_img.add(ee.Image.pixelLonLat().select(['latitude']))
                    .rename(['lat'])) \
        .addBands(mask_img.float().add(ee.Number(image.get('SOLAR_ZENITH_ANGLE')))
                    .rename(['sun_zenith'])) \
        .addBands(mask_img.float().add(ee.Number(image.get('SOLAR_AZIMUTH_ANGLE')))
                    .rename(['sun_azimuth'])) \
        .addBands(mask_img.add(1))


    return image


def getLAIforBiome(image, biome, rf_model):
    """
    Function that computes LAI for an input Landsat image and Random Forest models
    Args:
        image: ee.Image, must have training bands added
        biome: int
        rf_model: ee.Classifier
    """
    biom_lai = image.updateMask(image.select('biome2').eq(ee.Number(biome))) \
        .classify(rf_model, 'LAI')
    return biom_lai


def getLAIImage(image, sensor, nonveg):
    """
    Main Algorithm to computer LAI for a Landsat image
    Args:
        image:
        sensor: needs to be specified as a String ('LT05', 'LE07', 'LC08')
        nonveg: True if want to compute LAI for non-vegetation pixels
    """

    # Add necessary bands to image
    train_img = getTrainImg(image)

    # Start with an image of all zeros
    lai_img = train_img.select(['mask'], ['LAI']).multiply(0).double()

    if nonveg:
        biomes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    else:
        biomes = [1, 2, 3, 4, 5, 6, 7, 8]

    # Apply LAI for each biome
    for biome in biomes:
        lai_img = lai_img.where(
            train_img.select('biome2').eq(biome),
            getLAIforBiome(train_img, biome, getRFModel(sensor, biome)))

    # Set water LAI to zero
    water_mask = train_img.select('NDVI').lt(0) \
        .And(train_img.select('nir').lt(1000))

    lai_img = lai_img.where(water_mask, 0)
    qa = getLAIQA(train_img,sensor,lai_img)

    lai_img = lai_img.rename('LAI').multiply(100).round().clamp(0,65535).uint16()\
        .addBands(qa.byte())

    return ee.Image(lai_img.copyProperties(image)) \
        .set('system:time_start', image.get('system:time_start'))


def getLandsat(start, end, path, row):
    """
    Get Landsat image collection
    """
    # Landsat 8
    Landsat8_sr = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR') \
        .filterDate(start, end) \
        .filterMetadata('WRS_PATH','equals',path) \
        .filterMetadata('WRS_ROW','equals',row) \
        .filterMetadata('CLOUD_COVER','less_than',70) \
        .select(['B2','B3','B4','B5','B6','B7','pixel_qa'],
                ['blue','green','red','nir','swir1','swir2','pixel_qa']) \
        # .map(maskLST)
    
    # Landsat 7
    Landsat7_sr = ee.ImageCollection('LANDSAT/LE07/C01/T1_SR')  \
        .filterDate(start, end) \
        .filterMetadata('WRS_PATH','equals',path) \
        .filterMetadata('WRS_ROW','equals',row) \
        .filterMetadata('CLOUD_COVER','less_than',70) \
        .select(['B1','B2','B3','B4','B5','B7','pixel_qa'],
                ['blue','green','red','nir','swir1','swir2','pixel_qa']) \
        # .map(maskLST)

    # Landsat 5  
    Landsat5_sr = ee.ImageCollection('LANDSAT/LT05/C01/T1_SR') \
        .filterDate(start, end) \
        .filterMetadata('WRS_PATH','equals',path) \
        .filterMetadata('WRS_ROW','equals',row) \
        .filterMetadata('CLOUD_COVER','less_than',70) \
        .select(['B1','B2','B3','B4','B5','B7','pixel_qa'],
                ['blue','green','red','nir','swir1','swir2','pixel_qa']) \
        # .map(maskLST)
  
    Landsat_sr_coll = Landsat8_sr.merge(Landsat5_sr).merge(Landsat7_sr).map(setDate)

    return Landsat_sr_coll


def usage():
    usage_help = """
    Usage:
    python ee_Landsat_LAI_export.py -o <asset_dir> -p <path> -r <row> 
        -s <start_date> -d <end_date>
        
    Required arguments:
    -o  Earth Engine asset directory to export LAI images
    -p  WRS Path number of the Landsat Collection 1 surface reflectance scene
    -r  WRS Row number of the Landsat Collection 1 surface reflectance scene
    -s  The start date to export image in YYYY-MM-dd
    -e  The end date (exclusive) to export image in YYYY-MM-dd

    Optional arguments:
    -h  show this help
    -v  Boolean to indicate whether LAI is computed for non-vegetative (e.g. urban)
        pixels (based on NLCD). 1 - generate; 0 - do not generate. Default is 0.
     
    """
    print(usage_help)


def main(argv):
    
    path = 0
    row = 0
    start = ''
    end = ''
    assetDir = ''
    nonveg = 0

    try:
        opts, args = getopt.getopt(argv[1:],"ho:p:r:s:e:v:")
        if not opts:
            usage()
            sys.exit(2)
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt == '-o':
            print('assetDir: '+arg)
            assetDir = arg
        elif opt == '-p':
            print('WRS path: '+arg)
            path = int(arg)
        elif opt == '-r':
            print('WRS row: '+arg)
            row = int(arg)
        elif opt == '-s':
            print('start date: '+arg)
            start = arg
        elif opt == '-e':
            print('end date: '+arg)
            end = arg
        elif opt == '-v':
            print('nonveg: '+arg)
            nonveg = int(arg)

    # Set path row
    # path = int(argv[1])
    # row = int(argv[2])
    # start = argv[3]
    # end = argv[4]
    # nonveg = int(argv[5])

    # path = 30
    # row = 36
    # start = '2015-07-25'
    # end = '2015-07-26'
    pathrow = str(path).zfill(3)+str(row).zfill(3)
    # assetDir = 'users/yanghui/OpenET/LAI_US/test_LAI_maps/test_QA/scene_v0_1/'
    
    # Get Landsat collection
    landsat_coll = getLandsat(start, end, path, row)
    landsat_coll = landsat_coll.sort('system:time_start')
    
    n = landsat_coll.size().getInfo()
    print('Number of Landsat images: ', n)
    sys.stdout.flush()
    
    # print(laiColl.limit(10).getInfo())
    
    for i in range(n):
    
        landsat_image = ee.Image(landsat_coll.toList(5000).get(i))
        # print(landsat_image.getInfo())
    
        eedate = ee.Date(landsat_image.get('system:time_start'))
        date = eedate.format('YYYYMMdd').getInfo()
    
        sensor_dict = {'LANDSAT_5':'LT05','LANDSAT_7':'LE07','LANDSAT_8':'LC08'}
        sensor = landsat_image.get('SATELLITE').getInfo()
        sensor = sensor_dict[sensor]
    
        proj = landsat_image.select([0]).projection().getInfo()
        crs = proj['crs']
        transform = proj['transform']
    
        laiImage = getLAIImage(landsat_image, sensor, nonveg)
        
        laiImage = laiImage \
            .copyProperties(laiImage) \
            .set('system:time_start',laiImage.get('system:time_start')) \
            .set('lai_version',LAI_version)
        laiImage = ee.Image(laiImage)
    
        # date = laiImage.get('date')
        # outname = 'LAI_' + date.getInfo()
        # print(outname)
        outname = 'LAI_' + sensor + '_' + pathrow + '_' + date
    
    
        task = ee.batch.Export.image.toAsset(image = laiImage,
                                             description = outname,
                                             assetId = assetDir+outname,
                                             crs = crs,
                                             crsTransform = transform)
      
        
        task.start()  # submit task
        task_id = task.id
        print(task_id,outname)
        
        sys.stdout.flush()


if __name__ == "__main__":
    main(sys.argv)