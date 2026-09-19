function probe(crs,t) {
  var p=ee.Projection(crs,t),coords=pixelIndices(p,ee);
  function world(x,y){return [t[0]*x+t[1]*y+t[2],t[3]*x+t[4]*y+t[5]];}
  var region=ee.Geometry.Polygon([[world(0,0),world(4,0),world(4,4),world(0,4),world(0,0)]],crs,false);
  var x=coords.select('x'),y=coords.select('y');
  var valid=x.gte(0).and(x.lt(4)).and(y.gte(0)).and(y.lt(4)).and(x.eq(0).and(y.eq(0)).not());
  var raw=x.mod(2).rename('raw').updateMask(valid).reproject(p).clip(region);
  var cfg={stage:'projection-probe',maxPixels:1000,tileScale:1};
  print('WKT diagnostic only (never passed as crsTransform):',p.transform());
  var result=validateInputRaster(raw,cfg,ee);
  if(result.validPixels!==15)throw Error('Expected 15 valid pixels, got '+result.validPixels);
  try {validateInputRaster(raw.where(x.eq(1).and(y.eq(1)),2),cfg,ee);throw Error('Invalid encoding was accepted');}
  catch(e){if(String(e).indexOf('other than 0/1')<0)throw e;}
  try {validateInputRaster(raw.updateMask(ee.Image(0)),cfg,ee);throw Error('Empty input was accepted');}
  catch(e){if(String(e).indexOf('no valid pixels')<0)throw e;}
  print('PASS projection, binary values and NoData:',result);
}
