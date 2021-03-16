/**
 * Github: https://github.com/yanghuikang/Landsat-LAI
 * GEE: https://code.earthengine.google.com/?accept_repo=users/kangyanghui/LAI
 * Link: https://code.earthengine.google.com/87ff67124075111c9262af51b2938d5b
 *
 * Compute LAI for a Landsat image (full code standalone version)
 * 
 * LAI is generated using random forest and a training dataset derived 
 * from MODIS LAI and Landsat Collection 1 surface reflectance
 *
 * Author: Yanghui Kang
 * Contact: kangyanghui@gmail.com
 * 
 * Note: first time running may take a while, but after that the code
 * might get faster with the training set cached
 *
 * MAIN FUNCTION:
 *  var laiImage = getLAIImage(image, nonveg)
 *    Args:
 *      image: Landsat surface reflectance image (ee.Image)
 *      nonveg: Whether to computer LAI for non-vegetation pixels {Ture|False} 
 * 
 * OUTPUT:
 *  The output image has two bands
 *    LAI - LAI map scaled by 100 (scale factor = 0.01)
      QA  - LAI quality band    
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
 * 
 */ 


var LAI_version = '0.1.0';

// Rename Landsat bands
var renameLandsat = function(image) {
  var sensor = ee.String(image.get('SATELLITE'));
  var from = ee.Algorithms.If(sensor.compareTo('LANDSAT_8'),
                             ['B1','B2','B3','B4','B5','B7','pixel_qa'],
                             ['B2','B3','B4','B5','B6','B7','pixel_qa']);
  var to = ['blue','green','red','nir','swir1','swir2','pixel_qa'];

  return image.select(from, to);
};

// Return an image containing just the specified QA bits.
var getQABits = function(image, start, end, newName) {
    // Compute the bits we need to extract.
    var pattern = 0;
    for (var i = start; i <= end; i++) {
       pattern += Math.pow(2, i);
    }
    // Return a single band image of the extracted QA bits, giving the band
    // a new name.
    return image.select([0], [newName])
                  .bitwiseAnd(pattern)
                  .rightShift(start);
};

// Mask a Landsat image based on the QA band
var maskLST = function(image) {
  var pixelQA = image.select('pixel_qa');
  var cloud = getQABits(pixelQA, 1, 1, 'clear');
  return image.updateMask(cloud.eq(ee.Image(1)));
};

// Get crs transformation
var getAffineTransform = function(image) {
    var projection = image.projection();
    var json = ee.Dictionary(ee.Algorithms.Describe(projection));
    return ee.List(json.get('transform'));
};

// Add a QA band to the LAI image
var getLAIQA = function(landsat, sensor, lai) {
  /**
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
  */
  // minimum SR is always 0
  
  var red_max = 5100;
  var green_max = 5100;
  var nir_max = 7100;
  var swir1_max = 7100;
  var lai_max = 8;

  // information from the Landsat image
  var crs = landsat.select('red').projection().crs();
  var transform = getAffineTransform(landsat.select('red'));

  // Get pre-coded convex hull
  var data = ee.FeatureCollection('projects/ee-yanghuikang/assets/LAI/LAI_train_convex_hull_by_sensor_v0_1_1');

  var subset = data.filterMetadata('sensor','equals',sensor);
  subset = subset.sort('index');
  var hull_array = subset.aggregate_array('in_hull');
  var hull_array_reshape = ee.Array(hull_array).reshape([10,10,10,10]);

  // rescale landsat image
  var image_scaled = landsat.select('red').divide(red_max).multiply(10).floor().toInt()
    .addBands(landsat.select('green').divide(green_max).multiply(10).floor().toInt())
    .addBands(landsat.select('nir').divide(nir_max).multiply(10).floor().toInt())
    .addBands(landsat.select('swir1').divide(swir1_max).multiply(10).floor().toInt());

  // get an out-of-range mask
  var range_mask = landsat.select('red').gte(0)
    .and(landsat.select('red').lt(red_max))
    .and(landsat.select('green').gte(0))
    .and(landsat.select('green').lt(green_max))
    .and(landsat.select('nir').gte(0))
    .and(landsat.select('nir').lt(nir_max))
    .and(landsat.select('swir1').gte(0))
    .and(landsat.select('swir1').lt(swir1_max));

  // apply convel hull and get QA Band
  var hull_image = image_scaled.select('red').multiply(0)
    .add(ee.Image(hull_array_reshape)).updateMask(range_mask);

  var in_mask = hull_image
    .arrayGet(image_scaled.select(['red','green','nir','swir1'])
    .updateMask(range_mask));

  in_mask = in_mask.unmask(0).not().toByte();

  // check output range
  var out_mask = lai.gte(0).and(lai.lte(lai_max)).not().int();
  
  // indicate non-vegetation biome
  var biome_mask = landsat.select('biome2').eq(0).int();
  
  // combine
  var qa_band = in_mask.bitwiseOr(out_mask.leftShift(1)).bitwiseOr(biome_mask.leftShift(2)).toByte();
  // Map.addLayer(biome_mask.multiply(4),{},'biome mask');
  // Map.addLayer(qa_band,{},'qa');
  // print(in_mask,out_mask,biome_mask,qa_band);

  return qa_band.rename('QA');
};

// Compute VIs for an Landsat image
var getVIs = function(img) {
  
  var SR = img.expression('float(b("nir")) / b("red")');
  var NDVI = img.expression('float((b("nir") - b("red"))) / (b("nir") + b("red"))');
  var EVI = img.expression('2.5 * float((b("nir") - b("red"))) / (b("nir") + 6*b("red") - 7.5*float(b("blue")) + 10000)');
  // var GCI = img.expression('float(b("nir")) / b("green") - 1');
  // var EVI2 = img.expression('2.5 * float((b("nir") - b("red"))) / (b("nir") + 2.4*float(b("red")) + 10000)');
  // var OSAVI = img.expression('1.16 * float(b("nir") - b("red")) / (b("nir") + b("red") + 1600)');
  var NDWI = img.expression('float((b("nir") - b("swir1"))) / (b("nir") + b("swir1"))');
  // var NDWI2 = img.expression('float((b("nir") - b("swir2"))) / (b("nir") + b("swir2"))');
  // var MSR = img.expression('float(b("nir")) / b("swir1")');
  // var MTVI2 = img.expression('1.5*(1.2*float(b("nir") - b("green")) - 2.5*float(b("red") - b("green")))/sqrt((2*b("nir")+10000)*(2*b("nir")+10000) - (6*b("nir") - 5*sqrt(float(b("nir"))))-5000)');
  
  return img
    // .addBands(SR.select([0], ['SR']))
    .addBands(NDVI.select([0],['NDVI']))
    //.addBands(EVI.select([0],['EVI']))
    // .addBands(GCI.select([0],['GCI']))
    // .addBands(EVI2.select([0],['EVI2']))
    // .addBands(OSAVI.select([0],['OSAVI']))
    .addBands(NDWI.select([0],['NDWI']));
    // .addBands(NDWI2.select([0],['NDWI2']))
    // .addBands(MSR.select([0],['MSR']))
    // .addBands(MTVI2.select([0],['MTVI2']));
};

// Prepare feature bands for training
var getTrainImg = function(image) {
  
  // NLCD processing
  var year = ee.Date(image.get('system:time_start')).get('year').format('%d');

  var nlcd_dict = {
    '1997':'2001','1998':'2001','1999':'2001','2000':'2001','2001':'2001','2002':'2001',
    '2003':'2004','2004':'2004','2005':'2004',
    '2006':'2006','2007':'2006',
    '2008':'2008','2009':'2008',
    '2010':'2011','2011':'2011','2012':'2011',
    '2013':'2013','2014':'2013',
    '2015':'2016','2016':'2016','2017':'2016','2018':'2016','2019':'2016','2020':'2016','2021':'2016'};
  nlcd_dict = ee.Dictionary(nlcd_dict);
  var nlcd_year = nlcd_dict.get(year);
    
  var nlcd = ee.ImageCollection('USGS/NLCD')
    .filter(ee.Filter.eq('system:index',ee.String('NLCD').cat(nlcd_year)))
    .first();
  
  var fromList = [11,12,21,22,23,24,31,41,42,43,51,52,71,72,73,74,81,82,90,95];
  var toList = [0,0,0,0,0,0,0,1,2,3,0,4,5,0,0,0,5,6,7,8];
  var biome = ee.Image(nlcd).remap(fromList, toList);
  
  image = getVIs(image);

  // add other bands
  var mask_img = image.select(['pixel_qa'],['mask']).multiply(0);
  
  image = image.addBands(biome.rename(['biome2']))
    .addBands(mask_img.add(ee.Image.pixelLonLat()).select(['longitude'],['lon']))
    .addBands(mask_img.add(ee.Image.pixelLonLat()).select(['latitude'],['lat']))
    .addBands(mask_img.float().add(ee.Number(image.get('SOLAR_ZENITH_ANGLE'))).rename(['sun_zenith']))
    .addBands(mask_img.float().add(ee.Number(image.get('SOLAR_AZIMUTH_ANGLE'))).rename(['sun_azimuth']))
    .addBands(mask_img.add(1))
    .set('nlcd_year',nlcd_year);

  return image;
};

// Train an RF model for given biome and sensor
var getRFModel = function(sensor, biome) {
  
  var dir = 'projects/ee-yanghuikang/assets/LAI/LAI_train_sample_CONUS_v0_1_1';
  var train = ee.FeatureCollection(dir)
    .filterMetadata('sensor','equals',sensor);
  
  // change biome to client-site object
  if(biome>0) {
    train = train.filterMetadata('biome2','equals',biome);
  }

  // train
  var  features = ['red','green','nir','swir1','lat',
                   'lon','NDVI','NDWI','sun_zenith','sun_azimuth'];
  var rf = ee.Classifier.smileRandomForest({
      numberOfTrees: 100,
      minLeafPopulation: 50,
      variablesPerSplit: 5
    }).setOutputMode('REGRESSION')
    .train({
      features:train,
      classProperty: 'MCD_LAI',
      inputProperties: features
    });
    
  return rf;
};

// Computes LAI for a biome
var getLAIforBiome = function(image, rf_model, biome) {
  
  var biome_lai = image.updateMask(image.select('biome2').eq(ee.Number(biome)))
    .classify(rf_model,'LAI');
    
  return biome_lai;
};

// Computer LAI for a Landsat image
var getLAIImage = function(image, nonveg) {
  /** args:
    image: a Landsat image
    nonveg: whether to compute LAI for nonvegetative pixels (defined by NLCD)
            Defualt is False.
  */
  
  if(nonveg === null || nonveg === undefined) {
    nonveg = false;
  }
  
  // Rename landsat bands
  image = renameLandsat(image);
  var sensor_dict = ee.Dictionary({'5':'LT05','7':'LE07','8':'LC08'});
  var satellite = ee.String(image.get('SATELLITE')).slice(8,9);
  var sensor = sensor_dict.get(satellite);
  
  // Add feature bands to the image
  var train_img = getTrainImg(image);
  // print(train_img);
  
  // LAI for non-vegetative pixels is computed with samples from all biomes
  var biomes = [1,2,3,4,5,6,7,8];
  if(nonveg) {biomes = [0,1,2,3,4,5,6,7,8]}
  
  // Start with an image with fill value: 9999
  var lai_img = train_img.select(['mask'],['LAI']).multiply(0).add(9999).double();
  
  // Compute LAI for each biome
  for(var i in biomes){
    var biome = biomes[i];
    lai_img = lai_img.where(
      train_img.select('biome2').eq(biome),
      getLAIforBiome(train_img, getRFModel(sensor,biome), biome));
  }
  lai_img = lai_img.updateMask(lai_img.neq(9999));

  var qa = getLAIQA(train_img,sensor,lai_img);

  lai_img = lai_img.rename('LAI').multiply(100).round().clamp(0,65535).uint16()
    .addBands(qa.byte());
  
  return ee.Image(lai_img.copyProperties(image))
    .set('system:time_start',image.get('system:time_start'))
    .set('LAI_scale_factor',0.01)
    .set('LAI_NLCD_year',train_img.get('nlcd_year'))
    .set('LAI_nonveg',nonveg)
    .set('LAI_version',LAI_version);
};

// Get Landsat image collection
var getLandsat = function(start, end, region, cloud_limit) {
  
  if(cloud_limit === null || cloud_limit === undefined) {
    cloud_limit = 70;
  }
  
  var Landsat8_sr = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')  // Landsat 8
    .filterDate(start, end)
    .filterBounds(region)
    .filterMetadata('CLOUD_COVER','less_than',cloud_limit)
    .map(maskLST);
  // print(Landsat8_sr);
  
  var Landsat7_sr = ee.ImageCollection('LANDSAT/LE07/C01/T1_SR')  // Landsat 7
    .filterDate(start, end)
    .filterBounds(region)
    .filterMetadata('CLOUD_COVER','less_than',cloud_limit)
    .map(maskLST);
  // print(Landsat7_sr);
  
  var Landsat5_sr = ee.ImageCollection('LANDSAT/LT05/C01/T1_SR')  // Landsat 7
    .filterDate(start, end)
    .filterBounds(region)
    .filterMetadata('CLOUD_COVER','less_than',cloud_limit)
    .map(maskLST);
  
  var Landsat_sr_coll = Landsat8_sr.merge(Landsat7_sr).merge(Landsat5_sr)
    .sort('system:time_start');

  return Landsat_sr_coll;
};


// Set location and dates
var start = '2020-09-01';
var end = '2020-09-30';
var point = ee.Geometry.Point([-120.6200595561732, 37.12351640730834]);
var region = point.buffer(10000).bounds();

// Get Landsat image collection
var landsat_coll = getLandsat(start, end, point, 50);
print('landsat collection', landsat_coll);

var landsat_image = ee.Image(landsat_coll.first()).clip(region);
print('landsat_image',landsat_image);

var lai_image = getLAIImage(landsat_image, false);
print('lai_image',lai_image);

// Visualization
var visParam_false = {'min': 1000, 'max': [5000,6000,5000], 'bands': 'swir1,nir,red','gamma': 1.6};
var visParam_true = {'min': 0,'max': [2000,2000,2000],'bands':'red,green,blue'};
var visParam_LAI = {min:0.5,max:5,palette:["caa849","f4e87b","a7cc38","529d3b","248232","145b0b","1e4e18"]};

Map.addLayer(renameLandsat(landsat_image),visParam_false,'Landsat');
Map.addLayer(lai_image.select(['LAI']).divide(100),visParam_LAI,'LAI');
Map.addLayer(lai_image.select(['QA']),{min:0,max:5},'LAI-QA',false);
Map.addLayer(point,{},'location');
Map.centerObject(point,12);