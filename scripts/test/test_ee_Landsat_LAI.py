# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# import sys, getopt

try:
    import ee
except ModuleNotFoundError: 
    print('Please install earthengine-api before running this script.')
    
try:
    ee.Initialize()
except ee.EEException:
    ee.Authenticate()
    ee.Initialize()

LAI_version = '0.2.0'

ee.Initialize()

# Function to rename Landsat bands
def rename_landsat(image):
    spacecraft = ee.String(image.get('SPACECRAFT_ID'))
    spacecraft_no = ee.Number.parse(spacecraft.slice(8, 9))
    from_bands = ee.Algorithms.If(
        spacecraft_no.gte(8),
        ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL'],
        ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL']
    )
    to_bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'pixel_qa']
    return image.select(from_bands, to_bands).set('SPACECRAFT_NO', spacecraft_no)

# Function to scale Landsat data
def scale_landsat(image):
    return image.select(['green', 'red', 'nir', 'swir1']) \
                .multiply(0.0000275).add(-0.2).divide(0.0001) \
                .addBands(image.select('pixel_qa'))

# Function to mask Landsat data based on QA band
def mask_landsat(image):
    pixelQA = image.select('pixel_qa')
    def get_qa_bits(img, start, end, new_name):
        pattern = sum([2**i for i in range(start, end + 1)])
        return img.select([0], [new_name]).bitwiseAnd(pattern).rightShift(start)

    cloud = get_qa_bits(pixelQA, 3, 3, 'cloud')
    shadow = get_qa_bits(pixelQA, 4, 4, 'shadow')
    water = get_qa_bits(pixelQA, 7, 7, 'water')
    return image.updateMask(cloud.eq(ee.Image(0))) \
                .updateMask(shadow.eq(ee.Image(0))) \
                .updateMask(water.eq(ee.Image(0)))

# Function to compute vegetation indices
def compute_vis(image):
    NDVI = image.expression(
        'float((b("nir") - b("red"))) / (b("nir") + b("red"))'
    )
    NDWI = image.expression(
        'float((b("nir") - b("swir1"))) / (b("nir") + b("swir1"))'
    )
    return image.addBands(NDVI.rename('NDVI')).addBands(NDWI.rename('NDWI'))


# Function to process Landsat images
def process_landsat(image):
    renamed_image = rename_landsat(image)
    rescaled_image = scale_landsat(renamed_image)
    masked_image = mask_landsat(rescaled_image)
    final_image = compute_vis(masked_image)
    sun_elevation = ee.Number(image.get('SUN_ELEVATION'))
    sun_azimuth = ee.Number(image.get('SUN_AZIMUTH'))
    solar_zenith = ee.Algorithms.If(
        sun_elevation,
        ee.Number(90).subtract(sun_elevation),
        None
    )
    return ee.Image(final_image.copyProperties(image)).set({
        'SOLAR_AZIMUTH_ANGLE': sun_azimuth,
        'SOLAR_ZENITH_ANGLE': solar_zenith
    })

def getLandsat(start, end, path, row):
    """
    Get Landsat image collection
    """
    collections = [
        ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'),
        ee.ImageCollection('LANDSAT/LC08/C02/T1_L2'),
        ee.ImageCollection('LANDSAT/LE07/C02/T1_L2'),
        ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
    ]

    def setDate(image):
        """
        Function that adds a "date" property to an image in format "YYYYmmdd"
        """

        eeDate = ee.Date(image.get('system:time_start'))
        date = eeDate.format('YYYYMMdd')
        return image.set('date',date)

    landsat_sr_coll = collections[0]
    for collection in collections[1:]:
        landsat_sr_coll = landsat_sr_coll.merge(collection)
    # print(landsat_sr_coll.first().getInfo())
    
    landsat_sr_coll = landsat_sr_coll \
        .filterDate(start, end) \
        .filter(ee.Filter.lt('CLOUD_COVER',70)) \
        .filter(ee.Filter.eq('WRS_PATH',path)) \
        .filterMetadata('WRS_ROW','equals',row) \
        # .map(setDate)

    return landsat_sr_coll

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
        '2016': ['2015', '2016', '2017'],
        '2019': ['2018','2019','2020'],
        '2021': ['2021','2022','2023','2024','2025']
    }
    nlcd_dict = ee.Dictionary({
        src_year: tgt_year
        for tgt_year, src_years in nlcd_dict.items()
        for src_year in src_years})

    nlcd_year = nlcd_dict.get(
        ee.Date(image.get('system:time_start')).get('year').format('%d'))
    nlcd_year = ee.Number.parse(nlcd_year)

    # Function to extract the year from the NLCD image
    def set_nlcd_year(img):
        y = ee.Date(img.get('system:time_start')).get('year')
        return img.set('year', y)

    # NLCD collection and filtering
    nlcd_coll = ee.ImageCollection('USGS/NLCD_RELEASES/2019_REL/NLCD') \
        .merge(ee.ImageCollection('USGS/NLCD_RELEASES/2021_REL/NLCD')) \
        .map(set_nlcd_year)
    nlcd = nlcd_coll.filter(ee.Filter.eq('year', nlcd_year)).first()


    # Add the NLCD year as a property to track which year was used
    # image = image.set({'nlcd_year': nlcd_year})

    # Map NLCD codes to biomes
    nlcd_biom_remap = {
        11: 0, 12: 0,
        21: 0, 22: 0, 23: 0, 24: 0, 31: 0,
        41: 1, 42: 2, 43: 3, 52: 4,
        71: 5, 81: 5, 82: 6, 90: 7, 95: 8,
    }

    biom_img = ee.Image(nlcd).remap(*zip(*nlcd_biom_remap.items()))

    # Pre-process Landsat image: rename, rescale, cloud masking
    image = process_landsat(image)

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
        .addBands(mask_img.add(1)) \
        .set('nlcd_year',nlcd_year)

    return image


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


# def setDate(image):
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
    """

    filename = 'projects/ee-yanghuikang/assets/LAI/LAI_train_sample_CONUS_v0_1_1'

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

    scale_factor = 100
    lai_img = lai_img.rename('LAI').multiply(scale_factor).round().clamp(0,65535).uint16()\
        .addBands(qa.byte())

    return ee.Image(lai_img.copyProperties(image)) \
        .set('system:time_start', image.get('system:time_start')) \
        .set('LAI_scale_factor',1/scale_factor) \
        .set('LAI_nonveg',nonveg) \
        .set('LAI_NLCD_year',train_img.get('nlcd_year'))

#%% test loading Landsat images
# Get Landsat collection
start = '2024-04-18'
end = '2024-04-30'
path = 43
row = 33

landsat_coll = getLandsat(start, end, path, row)
landsat_coll = landsat_coll.sort('system:time_start')
# print(landsat_coll.getInfo())

# Get number of available Landsat (5/7/8/9) images
n = landsat_coll.size().getInfo()
print('Number of Landsat images: ', n)

print(landsat_coll.first().get('LANDSAT_PRODUCT_ID').getInfo())

#%% test process Landsat
test_image = ee.Image(landsat_coll.first())

renamed_image = rename_landsat(test_image)
print(renamed_image.getInfo()['bands'])

scaled_image = scale_landsat(renamed_image)
print(scaled_image.getInfo())

masked_image = mask_landsat(scaled_image)
print(masked_image.getInfo())

final_image = compute_vis(masked_image)
print(final_image.getInfo()['bands'])

processed_image = process_landsat(test_image)
print(processed_image.getInfo())

#%% test train image
train_image = getTrainImg(test_image)
print(train_image.bandNames().getInfo())

#%% test "getLAIImage()"
landsat_image = test_image
nonveg = 0
eedate = ee.Date(landsat_image.get('system:time_start'))
date = eedate.format('YYYYMMdd').getInfo()

# Landsat 9 will use Landsat 8 training set
sensor_dict = {'LANDSAT_5':'LT05','LANDSAT_7':'LE07','LANDSAT_8':'LC08','LANDSAT_9':'LC08'}
sensor = landsat_image.get('SPACECRAFT_ID').getInfo()
sensor = sensor_dict[sensor]

laiImage = getLAIImage(landsat_image, sensor, nonveg)

laiImage = laiImage \
    .copyProperties(laiImage) \
    .set('system:time_start',laiImage.get('system:time_start')) \
    .set('LAI_version',LAI_version)
laiImage = ee.Image(laiImage)

print(laiImage.getInfo())
