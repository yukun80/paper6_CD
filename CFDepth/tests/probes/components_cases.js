var grid={crs:'EPSG:4326',transform:[1/3600,0,110,0,-1/3600,30]};
var cfg={stage:'components-probe',maxPixels:1e7,tileScale:4,diagnosticDetails:true};
var p=ee.Projection(grid.crs,grid.transform),xy=pixelIndices(p,ee);
function regionAt(a){var t=grid.transform;
  return ee.Geometry.Rectangle([t[2]+a[0]*t[0],t[5]+a[3]*t[4],t[2]+a[2]*t[0],t[5]+a[1]*t[4]],grid.crs,false);
}
// 独立JavaScript BFS构造预期八连通编号，不调用生产矢量化或paint。
function reference(values,n){
  var labels=values.map(function(){return 0;}),minX=n,minY=n,maxX=-1,maxY=-1,groups=0;
  values.forEach(function(v,k){if(v){minX=Math.min(minX,k%n);maxX=Math.max(maxX,k%n);
    minY=Math.min(minY,Math.floor(k/n));maxY=Math.max(maxY,Math.floor(k/n));}});
  var width=maxX-minX+1;
  values.forEach(function(v,start){if(!v||labels[start])return;
    groups++;var queue=[start],members=[],minimum=Infinity;labels[start]=-1;
    for(var cursor=0;cursor<queue.length;cursor++){
      var k=queue[cursor],x=k%n,y=Math.floor(k/n);members.push(k);
      minimum=Math.min(minimum,(y-minY)*width+x-minX+1);
      for(var dy=-1;dy<=1;dy++)for(var dx=-1;dx<=1;dx++){
        var nx=x+dx,ny=y+dy,next=ny*n+nx;
        if(nx>=0&&nx<n&&ny>=0&&ny<n&&values[next]&&!labels[next]){labels[next]=-1;queue.push(next);}
      }
    }
    members.forEach(function(k){labels[k]=minimum;});
  });return {labels:labels,groups:groups};
}
function verify(name,support,expected,region,groups){
  var result=buildComponentRaster(support,grid,region,cfg,ee,function(m,v){print(name+' '+m,v);});
  var problem=componentAuditError(result.audit.counts);
  if(problem){showComponentFailure(result,region,cfg,ee,Map,print);throw Error(name+': '+problem);}
  var mismatch=componentIndicator(result.ids.unmask(0,false).neq(expected),p,ee);
  var errors=mismatch.rename('mismatch').reduceRegion(componentReductionArgs(grid,region,cfg,
    ee.Reducer.sum().unweighted())).get('mismatch').getInfo();
  if(errors!==0||result.audit.counts.vector_count!==groups)
    throw Error(name+': independent BFS/analytic component labels differ: '+errors);
  print('PASS '+name+': exact support AND independent component labels',result.audit.counts);
}
function small(name,n,fn,aoi){
  var values=[],keys=[];for(var k=0;k<n*n;k++){keys.push(k);values.push(fn(k%n,Math.floor(k/n))?1:0);}
  var ref=reference(values,n),region=regionAt(aoi||[0,0,n,n]);
  var key=xy.select('y').multiply(n).add(xy.select('x'));
  var inside=xy.select('x').gte(0).and(xy.select('x').lt(n)).and(xy.select('y').gte(0)).and(xy.select('y').lt(n));
  var support=key.remap(keys,values,0).where(inside.not(),0).rename('support').reproject(p).clip(region);
  var expected=key.remap(keys,ref.labels,0).where(inside.not(),0).reproject(p);
  verify(name,support,expected,region,ref.groups);
}
// 对全部12个网格位置做独立核对；不通过直方图或生产成员指示图验证自身。
function validateNullProbeSamples(features,labels){
  if (!Array.isArray(features)||features.length!==12)return 'Expected 12 sampled grid positions';
  var seen={},valid=0;
  for(var i=0;i<features.length;i++){
    var a=features[i].properties||{},x=a.column,y=a.row,key=x+','+y;
    if(typeof x!=='number'||typeof y!=='number'||x%1||y%1||x<0||x>=6||y<0||y>=2||seen[key])
      return 'Unexpected or duplicate grid coordinate: '+key;
    seen[key]=true;var wet=y===0&&x<5,expectedId=wet?labels[x]:0;
    if(!(a.support_mask>0)||a.support!==(wet?1:0)||!(a.painted_mask>0)||a.painted_cid!==expectedId)
      return 'Support or painted ID mismatch at '+key;
    if(wet){
      if(!(a.cid_mask>0)||a.cid!==expectedId)return 'Missing or wrong valid ID at '+key;
      valid++;
    }else if(a.cid_mask!==0||(a.cid!==null&&a.cid!==undefined))return 'Background ID is not masked at '+key;
  }
  return valid===5?null:'Expected five supported IDs';
}
function nullCountProbe(){
  var region=regionAt([0,0,6,2]),x=xy.select('x'),y=xy.select('y');
  var support=x.gte(0).and(x.lt(5)).and(y.eq(0)).rename('support').reproject(p).clip(region);
  var labels=[210,220,230,240,502049];
  var painted=x.remap([0,1,2,3,4],labels,0).where(y.neq(0),0).rename('cid').reproject(p);
  var ids=painted.updateMask(support).reproject(p);
  var sampleImage=xy.rename(['column','row'])
    .addBands(ee.Image.pixelCoordinates(p).rename(['raw_x','raw_y'])).addBands(support)
    .addBands(support.mask().rename('support_mask'))
    .addBands(painted.rename('painted_cid')).addBands(painted.mask().rename('painted_mask'))
    .addBands(ids).addBands(ids.mask().rename('cid_mask'));
  var samples=sampleImage.sample({region:region,projection:p,dropNulls:false,geometries:false,tileScale:cfg.tileScale});
  var sampleInfo=samples.getInfo();
  var inputProblem=validateNullProbeSamples(sampleInfo.features,labels);
  print('Null probe input grid and independent sample check:',{grid:grid,error:inputProblem});
  if(inputProblem){
    print('Null probe all grid positions, including masked values:',samples);
    throw Error('Null probe input failed: '+inputProblem);
  }
  var audit=auditComponentRaster(support,ids,painted,grid,region,cfg,ee,labels);
  audit.counts.vector_count=labels.length;
  var problem=componentAuditError(audit.counts);
  print('GEE null-count comparison (ordinary vs non-null):',audit.counts);
  var eachOnce=labels.every(function(id){return audit.counts.raster_id_pixels[id]===1;});
  if(problem||audit.counts.expected!==5||audit.counts.restored!==5||audit.counts.distinct_ids!==5||!eachOnce){
    print('Null probe all grid positions, including masked values:',samples);
    throw Error('Null-count probe failed: '+(problem||'Expected five supported IDs, each occurring once'));
  }
  // 普通计数仅作观测，不强制所有GEE后端均必须计入null；正式条件始终为非空计数及集合一致。
  print('PASS five valid IDs plus masked background');
}
