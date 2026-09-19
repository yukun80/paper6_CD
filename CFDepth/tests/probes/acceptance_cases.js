// 汇总独立检查；服务端异常记FAIL，依赖失败记SKIP，不以未执行冒充成功。
var acceptanceResults=[],acceptanceStatus={},acceptanceCache={};
var probeGroups={
  quick:['coordinates_geographic','coordinates_projected','coordinates_rotated','mixed_reducers','null_and_five_ids'],
  geometry:['coordinates_geographic','coordinates_projected','coordinates_rotated','physical_centers_geographic','physical_centers_projected','physical_centers_rotated','projection_and_binary_geographic','projection_and_binary_projected','projection_and_binary_rotated'],
  topology:['coordinates_geographic','topology_non_aligned','topology_hole','topology_diagonal_singleton','topology_independent','topology_small_fixture','relative_component_ids'],
  prepare:['coordinates_geographic','null_and_five_ids','production_components','production_prepare_and_groups'],
  solver:['coordinates_geographic','four_colors_and_neighbors','compact_solver_input','non_equilibrium_real_sweep','compact_state_and_rejection'],
  gradient:['coordinates_geographic','analytic_gradients_and_masks'],
  assets:['saved_stage_contracts']
};
if(!probeGroups[PROBE_GROUP])throw Error('Unknown PROBE_GROUP: '+PROBE_GROUP);

function assertProbe(ok,message){if(!ok)throw Error(message);}
function runCheck(name,dependencies,body){
  if(probeGroups[PROBE_GROUP].indexOf(name)<0){
    acceptanceStatus[name]='NOT_RUN';acceptanceResults.push({name:name,status:'NOT_RUN',elapsed_ms:0});return;
  }
  var started=Date.now();
  var blocked=dependencies.filter(function(key){return acceptanceStatus[key]!=='PASS';});
  if(blocked.length){
    acceptanceStatus[name]='SKIP';acceptanceResults.push({name:name,status:'SKIP',elapsed_ms:0,reason:'Blocked by '+blocked.join(', ')});
    print('SKIP '+name,blocked);return;
  }
  try{body();acceptanceStatus[name]='PASS';acceptanceResults.push({name:name,status:'PASS',elapsed_ms:Date.now()-started});print('PASS '+name);}
  catch(error){var message=String(error);acceptanceStatus[name]='FAIL';acceptanceResults.push({name:name,status:'FAIL',elapsed_ms:Date.now()-started,reason:message});print('FAIL '+name,message);}
}
function maxValue(image,g){return g.total(image.rename('v'),ee.Reducer.max().unweighted()).get('v').getInfo();}
function pixelCount(mask,g){return g.total(componentIndicator(mask,g.P,ee).rename('n'),ee.Reducer.sum().unweighted()).get('n').getInfo();}
function nearValue(actual,expected,tolerance,message){assertProbe(typeof actual==='number'&&isFinite(actual)&&Math.abs(actual-expected)<=tolerance,message+': '+actual+' vs '+expected);}
function validatePacked(image,kind,grid){
  var info=packStage(image,kind).getInfo();
  validateStageBands({grid_crs:grid.crs,grid_transform:grid.transform},info.bands,kind,'acceptance '+kind);
  assertProbe(info.bands.length===stageBandNames(kind).length,'Unexpected saved band count');
}
// 几何变换只服务于物理中心检查，不作为整数索引检查的隐含前置条件。
function coordinateSamples(crs,t,includeWorld){
  var projection=ee.Projection(crs,t),raw=ee.Image.pixelCoordinates(projection),indices=pixelIndices(projection,ee);
  function world(x,y){return [t[0]*x+t[1]*y+t[2],t[3]*x+t[4]*y+t[5]];}
  var region=ee.Geometry.Polygon([[world(-2,-2),world(2,-2),world(2,2),world(-2,2),world(-2,-2)]],crs,false);
  var samples=indices.rename(['column','row']).addBands(raw.rename(['raw_x','raw_y']))
    .sample({region:region,projection:projection,geometries:includeWorld,dropNulls:false});
  if(includeWorld){
    // 少量采样点使用合法的零误差；不改变生产区域的几何误差或求解容差。
    samples=samples.map(function(f){return f.set('world',
      f.geometry().transform(crs,ee.ErrorMargin(0,'meters')).coordinates());});
  }
  return samples.getInfo().features;
}
function validateCoordinateIndices(samples){
  assertProbe(Array.isArray(samples)&&samples.length===16,'Expected 16 positive/negative coordinate samples');
  var seen={};
  samples.forEach(function(f){
    var a=f.properties,x=a.column,y=a.row,key=x+','+y;
    assertProbe(typeof x==='number'&&typeof y==='number'&&x%1===0&&y%1===0&&x>=-2&&x<2&&y>=-2&&y<2&&!seen[key],'Invalid integer indices '+key);
    seen[key]=true;
    assertProbe(typeof a.raw_x==='number'&&isFinite(a.raw_x)&&typeof a.raw_y==='number'&&isFinite(a.raw_y)&&
      x===Math.floor(a.raw_x)&&y===Math.floor(a.raw_y),'Wrong center-to-index conversion');
  });
}
function coordinateIndexProbe(crs,t){
  var samples=coordinateSamples(crs,t,false);
  validateCoordinateIndices(samples);
  print('Coordinates: 16 index samples passed',CONFIG.diagnosticDetails?samples:crs);
}
function coordinateCenterProbe(crs,t){
  var samples=coordinateSamples(crs,t,true);
  validateCoordinateIndices(samples);
  samples.forEach(function(f){
    var a=f.properties,x=a.column+0.5,y=a.row+0.5;
    var expected=[t[0]*x+t[1]*y+t[2],t[3]*x+t[4]*y+t[5]];
    var tol=Math.max(Math.abs(t[0])+Math.abs(t[1]),Math.abs(t[3])+Math.abs(t[4]))*1e-5;
    assertProbe(Array.isArray(a.world)&&a.world.length===2,'Missing physical point coordinates');
    nearValue(a.world[0],expected[0],tol,'Physical pixel center x');nearValue(a.world[1],expected[1],tol,'Physical pixel center y');
  });
  print('Coordinates: 16 physical centers passed',CONFIG.diagnosticDetails?samples:crs);
}
function summarizeAcceptance(results){
  var totals={total:results.length,passed:0,failed:0,skipped:0,not_run:0};
  results.forEach(function(r){
    if(r.status==='PASS')totals.passed++;
    else if(r.status==='FAIL')totals.failed++;
    else if(r.status==='SKIP')totals.skipped++;
    else if(r.status==='NOT_RUN')totals.not_run++;
    else throw Error('Unknown acceptance status: '+r.status);
  });
  return totals;
}
// sum保留原区域权重；极值参考检查所有有原始掩膜的相交像元。
function reducerReferenceBounds(bounds,method){
  return method==='sum'?bounds.slice():[Math.floor(bounds[0]),Math.floor(bounds[1]),Math.ceil(bounds[2]),Math.ceil(bounds[3])];
}
function mixedReducerProbe(){
  var names=['a','peak','b','floor'],methods=['sum','max','sum','min'],requests={};
  var definitions=[{key:'aligned',bounds:[0,0,4,2],groups:[1,2]},
    {key:'non_aligned',bounds:[.2,.2,3.2,2.2],groups:[1]}];
  definitions.forEach(function(def){
    var region=regionAt(def.bounds),g=createGEEGrid(grid,region,CONFIG,ee),x=g.xy().select('x'),y=g.xy().select('y');
    var ids=g.C(1).rename('cid'),mask=g.C(1).where(x.eq(1),.4);
    if(def.key==='aligned'){
      ids=ids.where(x.eq(2),2).where(x.eq(3),0);
      mask=mask.where(x.eq(0).and(y.eq(1)),0);
    }
    var support=ids.gt(0).rename('support');
    var D=g.pack([g.C(0).rename('dem'),ids,support,g.C(0).rename('dry'),support.rename('hard')]);
    var im=g.pack([x.add(1).rename('a'),x.multiply(10).add(7).rename('peak'),
      x.add(2).multiply(3).rename('b'),x.subtract(8).rename('floor')]).updateMask(mask);
    var solver=createGEESolver(D,grid,region,CONFIG,ee),reference={};
    def.groups.forEach(function(id){
      reference[id]={};
      // 非对齐单分量保留原参考输入；两分量用where清零其他成员的原掩膜，独立于生产helper。
      var ref=def.key==='non_aligned'?im:im.updateMask(im.mask().where(ids.neq(id).unmask(1,false),0));
      names.forEach(function(n,i){
        var args=g.geometryArgs(ee.Reducer[methods[i]]());
        if(methods[i]!=='sum')args.geometry=regionAt(reducerReferenceBounds(def.bounds,methods[i]));
        reference[id][n]=ref.select(n).reduceRegion(args).get(n);
      });
    });
    var sorted=groupedLayout(names,methods,'probe'),input=groupedInput(im,sorted.names,ids,ee,g.P);
    var sample=g.xy().rename(['column','row']).addBands(ids.rename('source_cid').unmask(0,false))
      .addBands(im.mask().rename(names.map(function(n){return n+'_before';})))
      .addBands(input.select(names).mask().rename(names.map(function(n){return n+'_after';})))
      .addBands(input.select('cid').mask().rename('cid_mask'))
      // 用GEE实际编码的0.4掩膜作对照，避免JS双精度字面量与服务端掩膜表示差异。
      .addBands(ee.Image.constant(1).updateMask(ee.Image.constant(.4)).mask().rename('fraction_control'));
    requests[def.key]={grouped:solver.grouped(im,names,methods).toList(10),reference:reference,
      samples:sample.sample({region:region,projection:g.P,dropNulls:false,geometries:false}).toList(20),
      legacy_a:g.total(im.select('a').updateMask(ids.gt(0)),ee.Reducer.sum()).get('a')};
    if(def.key==='non_aligned'){
      // 第4列与区域相交但中心在外；第5列完全在外。只读取这两个诊断像元。
      var boundarySample=g.xy().rename(['column','row'])
        .addBands(im.select('peak').unmask(0,false))
        .addBands(im.select('peak').mask().rename('original_mask'))
        .addBands(input.select('peak').mask().rename('filtered_mask'));
      requests[def.key].boundary_samples=boundarySample.sample({region:regionAt([3,0,5,1]),
        projection:g.P,dropNulls:false,geometries:false}).toList(3);
    }
  });
  // 所有字段及像元先一次取回并打印，再汇总断言，不在首个字段失败时隐藏后续结果。
  var results=ee.Dictionary(requests).getInfo(),errors=[];
  print('Mixed reducers: all fields, original/filtered masks and legacy a (diagnostic only)',results);
  function equal(a,b,message){if(typeof a!=='number'||typeof b!=='number'||!isFinite(a)||!isFinite(b)||Math.abs(a-b)>1e-9)errors.push(message+': '+a+' vs '+b);}
  definitions.forEach(function(def){
    var result=results[def.key],seen={};
    result.grouped.forEach(function(f){var row=f.properties,id=row.cid;
      if(seen[id]||def.groups.indexOf(id)<0){errors.push(def.key+' unexpected/duplicate cid '+id);return;}seen[id]=true;
      names.forEach(function(n){equal(row[n],result.reference[id][n],def.key+' cid='+id+' field='+n);});
      if(def.key==='aligned'){
        var analytic=id===1?{peak:17,floor:-8}:{peak:27,floor:-6};
        ['peak','floor'].forEach(function(n){equal(row[n],analytic[n],'Aligned analytic '+id+' '+n);});
      }
    });
    def.groups.forEach(function(id){if(!seen[id])errors.push(def.key+' missing cid '+id);});
    if(def.key==='non_aligned'){
      var boundary=result.boundary_samples||[],boundarySeen={};
      if(boundary.length!==2)errors.push('Expected two boundary diagnostic pixels');
      boundary.forEach(function(f){var a=f.properties,key=a.column+','+a.row;
        if(boundarySeen[key]||a.row!==0||[3,4].indexOf(a.column)<0){errors.push('Invalid boundary coordinate '+key);return;}
        boundarySeen[key]=true;
        if(a.column===3){
          equal(a.peak,37,'Intersecting boundary peak');
          if(!(a.original_mask>0))errors.push('Intersecting boundary lost original mask');
          equal(a.filtered_mask,a.original_mask,'Boundary fractional mask');
          equal(result.reference[1].peak,a.peak,'Reference must include intersecting boundary');
        }else{
          equal(a.original_mask,0,'Exterior original mask');equal(a.filtered_mask,0,'Exterior filtered mask');
        }
      });
      if(!boundarySeen['3,0']||!boundarySeen['4,0'])errors.push('Missing boundary/exterior sample');
    }
    var expectedCount=def.key==='aligned'?8:6,positions={};
    // 非对齐区域按像元中心采样，不能用加权覆盖的12个交叠像元数作为样本数。
    if(result.samples.length!==expectedCount)errors.push(def.key+' sample count '+result.samples.length+' vs '+expectedCount);
    result.samples.forEach(function(f){var a=f.properties,key=a.column+','+a.row;
      if(positions[key])errors.push(def.key+' duplicate sample '+key);positions[key]=true;
      var admitted=a.source_cid>0;
      names.forEach(function(n){equal(a[n+'_after'],admitted?a[n+'_before']:0,def.key+' mask '+key+' '+n);});
      equal(a.cid_mask,admitted?1:0,def.key+' cid gate '+key);
      if(def.key==='aligned'){
        if(!(a.fraction_control>0&&a.fraction_control<1))errors.push('Invalid fractional mask control');
        var expected=a.column===0&&a.row===1?0:a.column===1?a.fraction_control:1;
        names.forEach(function(n){equal(a[n+'_before'],expected,'Aligned original mask '+key+' '+n);});
      }
    });
  });
  if(errors.length)throw Error('Mixed reducer/mask checks failed: '+errors.join('; '));
  print('PASS fractional masks, component isolation and all mixed reducer fields');
}
runCheck('coordinates_geographic',[],function(){coordinateIndexProbe('EPSG:4326',grid.transform);});
runCheck('coordinates_projected',[],function(){coordinateIndexProbe('EPSG:3857',[10,0,1918543,0,-10,7241630]);});
runCheck('coordinates_rotated',[],function(){coordinateIndexProbe('EPSG:3857',[10,2,1918543,1,-10,7241630]);});
runCheck('physical_centers_geographic',['coordinates_geographic'],function(){coordinateCenterProbe('EPSG:4326',grid.transform);});
runCheck('physical_centers_projected',['coordinates_projected'],function(){coordinateCenterProbe('EPSG:3857',[10,0,1918543,0,-10,7241630]);});
runCheck('physical_centers_rotated',['coordinates_rotated'],function(){coordinateCenterProbe('EPSG:3857',[10,2,1918543,1,-10,7241630]);});
runCheck('projection_and_binary_geographic',['coordinates_geographic'],function(){probe('EPSG:4326',grid.transform);});
runCheck('projection_and_binary_projected',['coordinates_projected'],function(){probe('EPSG:3857',[9.32480563590904,0,1918543.3940000013,0,-9.324805635909165,7241630.245534113]);});
runCheck('projection_and_binary_rotated',['coordinates_rotated'],function(){probe('EPSG:3857',[10,2,1918543,1,-10,7241630]);});
runCheck('mixed_reducers',[],mixedReducerProbe);
runCheck('null_and_five_ids',['coordinates_geographic'],nullCountProbe);
runCheck('topology_non_aligned',['coordinates_geographic'],function(){small('non-grid-aligned boundary',6,function(x,y){return x<4&&y<4;},[0.2,0.2,5.2,5.2]);});
runCheck('topology_hole',['coordinates_geographic'],function(){small('hole',9,function(x,y){return x>0&&x<8&&y>0&&y<8&&!(x>=3&&x<=5&&y>=3&&y<=5);});});
runCheck('topology_diagonal_singleton',['coordinates_geographic'],function(){small('diagonal and singleton',8,function(x,y){return (x===1&&y===1)||(x===2&&y===2)||(x===6&&y===6);});});
runCheck('topology_independent',['coordinates_geographic'],function(){small('one pixel gap',9,function(x,y){return y>=2&&y<=6&&((x>=1&&x<=3)||(x>=5&&x<=7));});});
runCheck('topology_small_fixture',['coordinates_geographic'],function(){
  small('existing small fixture topology',36,function(x,y){return (x>=4&&x<24&&y>=4&&y<24&&!(x===14&&y===14))||
    (x>=25&&x<29&&y>=25&&y<29)||(x===27&&y===32)||(x===28&&y===33)||(x===34&&y===34);});
});
runCheck('relative_component_ids',['coordinates_geographic'],function(){
  // 原始坐标的统一中心偏移应在减去最小值时抵消；与整数索引生成的ID逐像元比较。
  var region=regionAt([-2,-2,3,3]),raw=ee.Image.pixelCoordinates(p),idx=pixelIndices(p,ee);
  var ext=raw.reduceRegion({reducer:ee.Reducer.minMax().unweighted(),geometry:region,crs:grid.crs,crsTransform:grid.transform,maxPixels:100});
  var rawKey=raw.select('y').subtract(ee.Number(ext.get('y_min'))).multiply(5)
    .add(raw.select('x').subtract(ee.Number(ext.get('x_min')))).add(1);
  var key=idx.select('y').add(2).multiply(5).add(idx.select('x').add(2)).add(1);
  nearValue(rawKey.subtract(key).abs().reduceRegion({reducer:ee.Reducer.max().unweighted(),geometry:region,crs:grid.crs,crsTransform:grid.transform,maxPixels:100}).values().get(0).getInfo(),0,0,'Relative component ID changed');
});
runCheck('four_colors_and_neighbors',['coordinates_geographic'],function(){
  var region=regionAt([-4,-4,4,4]),g=createGEEGrid(grid,region,CONFIG,ee),idx=g.xy(),colors=fourColors(idx);
  var memberships=g.C(0);for(var k=0;k<4;k++)memberships=memberships.add(colors.eq(k));
  nearValue(maxValue(memberships.neq(1),g),0,0,'Some pixels are assigned no/multiple colors');
  nearValue(maxValue(colors.mod(1).neq(0).or(colors.lt(0)).or(colors.gt(3)),g),0,0,'Colors outside integer 0..3');
  g.dirs.forEach(function(d){nearValue(maxValue(colors.eq(g.at(colors,d[0],d[1])),g),0,0,'Adjacent pixels share a color');});
  nearValue(maxValue(g.at(idx.select('x'),1,0).subtract(idx.select('x').add(1)).abs(),g),0,0,'Horizontal neighbor direction');
  nearValue(maxValue(g.at(idx.select('y'),0,1).subtract(idx.select('y').add(1)).abs(),g),0,0,'Vertical neighbor direction');
});
runCheck('production_components',['coordinates_geographic','null_and_five_ids'],function(){
  var localCfg={};Object.keys(CONFIG).forEach(function(k){localCfg[k]=CONFIG[k];});
  // 只缩短探针扫描阶段，不改变正式CONFIG/资产参数。
  localCfg.sweepsPerStage=1;localCfg.maxPixels=1e6;localCfg.stage='components';
  var region=regionAt([0,0,32,16]),g=createGEEGrid(grid,region,localCfg,ee),ij=g.xy(),x=ij.select('x'),y=ij.select('y');
  var patchA=x.gte(4).and(x.lt(10)).and(y.gte(4)).and(y.lt(10));
  var patchB=x.gte(20).and(x.lt(26)).and(y.gte(4)).and(y.lt(10));
  var isolated=x.eq(28).and(y.eq(12)),support=patchA.or(patchB).or(isolated).reproject(g.P).clip(region);
  var raw=support.toByte().rename('raw'),dem=g.C(100).where(support,99).rename('dem');
  var built=buildComponentStage(raw,dem,grid,region,localCfg,ee,print);
  nearValue(built.audit.audit.counts.expected,73,0,'Production support count');
  nearValue(built.audit.audit.counts.distinct_ids,3,0,'Production component count');
  validatePacked(built.image,'components',grid);
  acceptanceCache={cfg:localCfg,region:region,g:g,D:built.image,patchA:patchA,patchB:patchB,isolated:isolated};
});
runCheck('production_prepare_and_groups',['production_components'],function(){
  var a=acceptanceCache,solver=createGEESolver(a.D,grid,a.region,a.cfg,ee),result=solver.prepare();
  validatePacked(result.image,'prepared',grid);
  var S=result.image.select('S');
  a.prepared=result.image;a.solver=createGEESolver(a.D,grid,a.region,a.cfg,ee,a.prepared);
  var table=a.solver.grouped(a.D.select('support').rename('n'),['n'],['sum']);
  var broadcast=a.solver.broadcast(table,'n');
  var checks=ee.Dictionary({rows:result.table.toList(3),
    initial_error:a.g.total(S.updateMask(a.patchA.or(a.patchB)).subtract(99.5).abs().rename('v'),ee.Reducer.max()).get('v'),
    eligible_pixels:a.g.total(componentIndicator(result.image.select('eligible'),a.g.P,ee).rename('n'),ee.Reducer.sum().unweighted()).get('n'),
    broadcast_error:a.g.total(broadcast.updateMask(a.patchA.or(a.patchB)).subtract(36).abs().rename('v'),ee.Reducer.max()).get('v')}).getInfo();
  var rows=checks.rows;assertProbe(rows.length===3,'Preparation omitted a component');
  var eligible=0;rows.forEach(function(f){var r=f.properties;
    ['cid','anchors','initial','weight_sum','weighted_mid'].forEach(function(key){assertProbe(typeof r[key]==='number'&&isFinite(r[key]),'Non-finite group field '+key);});
    if(r.eligible)eligible++;
  });
  assertProbe(eligible===2,'Two patches should be eligible; singleton must be unsupported');
  nearValue(checks.initial_error,0,1e-8,'Prepared midpoint initial WSE');
  nearValue(checks.eligible_pixels,72,0,'Eligible pixels');
  nearValue(checks.broadcast_error,0,1e-8,'Grouped count/broadcast');
});
// 显式小图仅验证扫描/诊断，不展开生产prepare和多阶段求解；prepare另组独立验收。
runCheck('compact_solver_input',['coordinates_geographic'],function(){
  var cfg={};Object.keys(CONFIG).forEach(function(k){cfg[k]=CONFIG[k];});
  var region=regionAt([0,0,12,6]),g=createGEEGrid(grid,region,cfg,ee),x=g.xy().select('x'),y=g.xy().select('y');
  var patchA=x.gte(1).and(x.lt(4)).and(y.gte(1)).and(y.lt(5));
  var patchB=x.gte(7).and(x.lt(10)).and(y.gte(1)).and(y.lt(5));
  var support=patchA.or(patchB).clip(region),ids=patchA.add(patchB.multiply(2)).rename('cid');
  var D=packStage(g.pack([g.C(99).rename('dem'),ids,support.rename('support'),support.not().rename('dry'),support.rename('hard')]),'components');
  var prepared=D.addBands(g.pack([g.C(99.25).rename('lower'),g.C(100).rename('upper'),g.C(99.5).rename('mid'),g.C(.8).rename('weight'),support.rename('eligible'),g.C(99.5).rename('S'),g.C(99.5).rename('baseS')]));
  STATE_FIELDS.forEach(function(k){prepared=prepared.addBands(g.C(k==='status'?1:0).rename(k));});
  prepared=packStage(prepared.updateMask(support),'prepared');
  acceptanceCache={cfg:cfg,region:region,g:g,D:D,patchA:patchA,patchB:patchB,prepared:prepared,
    solver:createGEESolver(D,grid,region,cfg,ee,prepared)};
  validatePacked(prepared,'prepared',grid);
});
runCheck('non_equilibrium_real_sweep',['compact_solver_input','four_colors_and_neighbors'],function(){
  var a=acceptanceCache,g=a.g,S=a.prepared.select('S');
  var perturbed=S.add(fourColors(g.xy()).eq(0).and(a.patchA).multiply(0.2));
  var active=a.prepared.select('eligible'),next=a.solver.sweep(perturbed,g.C(0),active);
  var checked=ee.Dictionary({
    changed:g.total(next.subtract(perturbed).abs().rename('v'),ee.Reducer.max()).get('v'),
    before:g.total(a.solver.energyImages(perturbed,g.C(0)).select('primary'),ee.Reducer.sum()).get('primary'),
    after:g.total(a.solver.energyImages(next,g.C(0)).select('primary'),ee.Reducer.sum()).get('primary'),
    hard:g.total(a.D.select('dem').add(CONFIG.minDepth).subtract(next).max(0).updateMask(a.D.select('hard')).rename('v'),ee.Reducer.max()).get('v'),
    independent:g.total(next.subtract(S).abs().updateMask(a.patchB).rename('v'),ee.Reducer.max()).get('v')}).getInfo();
  assertProbe(checked.changed>1e-6,'Solver did not update non-equilibrium input');
  assertProbe(isFinite(checked.before)&&isFinite(checked.after)&&checked.after<checked.before,'One real sweep did not reduce the primary objective');
  nearValue(checked.hard,0,1e-9,'Hard lower bound');nearValue(checked.independent,0,1e-8,'Independent component changed');
  print('Real sweep checks',checked);
});
runCheck('compact_state_and_rejection',['compact_solver_input'],function(){
  var a=acceptanceCache,state=packStage(a.prepared,'state');validatePacked(state,'state',grid);
  var rows=a.solver.diagnostics(state.select('S'),a.g.C(0),state).getInfo().features;
  assertProbe(rows.length===2,'Compact diagnostics lost a component');
  rows.forEach(function(f){nearValue(f.properties.bad,0,0,'Non-finite state');nearValue(f.properties.residual,0,1e-8,'Analytic residual');});
  var rejected=false;
  try{a.solver.finalize(state);}catch(e){if(String(e).indexOf('Unfinished components')<0)throw e;rejected=true;}
  assertProbe(rejected,'Unfinished states were allowed to publish');
  var unsupported=state.addBands(a.g.C(0).rename('status'),null,true);
  nearValue(pixelCount(a.solver.finalize(unsupported).valid,a.g),0,0,'Unsupported state published');
});
runCheck('analytic_gradients_and_masks',['coordinates_geographic'],function(){
  // 梯度使用实际生产距离和差分路径，期望值从独立中心几何距离计算。
  [0,60].forEach(function(latitude){
    var gr={crs:'EPSG:4326',transform:[1/3600,0,110,0,-1/3600,latitude]},pp=ee.Projection(gr.crs,gr.transform);
    var region=ee.Geometry.Rectangle([110,latitude-8/3600,110+8/3600,latitude],gr.crs,false);
    var g=createGEEGrid(gr,region,CONFIG,ee),ij=g.xy(),x=ij.select('x'),y=ij.select('y');
    var valid=x.gte(1).and(x.lt(7)).and(y.gte(1)).and(y.lt(7)).clip(region);
    var D=packStage(g.pack([g.C(0).rename('dem'),valid.rename('cid'),valid.rename('support'),valid.not().rename('dry'),valid.rename('hard')]),'components');
    var solver=createGEESolver(D,gr,region,CONFIG,ee),plane=x.multiply(2).add(y.multiply(3)).toDouble().updateMask(valid);
    var result=solver.gradientImages(plane,valid);
    nearValue(pixelCount(result.gradient.mask(),g),36,0,'Planar gradient coverage');
    nearValue(pixelCount(result.hasX.and(result.hasY).and(valid),g),36,0,'Planar two-direction coverage');
    var lat=ee.Image.pixelLonLat().select('latitude').reproject(pp).multiply(Math.PI/180);
    var dx=lat.cos().multiply(EARTH_RADIUS*Math.PI/180/3600),dy=EARTH_RADIUS*Math.PI/180/3600;
    var expected=dx.pow(-1).multiply(2).pow(2).add(Math.pow(3/dy,2)).sqrt();
    nearValue(maxValue(result.gradient.subtract(expected).abs(),g),0,1e-9,'Planar WSE magnitude including one-sided edges');
    var row=x.gte(1).and(x.lt(7)).and(y.eq(2)).clip(region);
    var line=solver.gradientImages(plane.updateMask(row),row);
    nearValue(pixelCount(line.hasX.and(line.hasY).and(row),g),0,0,'One-row surface has two observed directions');
    nearValue(pixelCount(line.gradient.mask(),g),6,0,'One-row gradient coverage');
    nearValue(maxValue(line.gradient.subtract(dx.pow(-1).multiply(2)).abs(),g),0,1e-9,'One-direction gradient');
    var single=x.eq(2).and(y.eq(2)).clip(region),isolated=solver.gradientImages(plane.updateMask(single),single);
    nearValue(pixelCount(isolated.gradient.mask(),g),0,0,'Isolated pixel has a gradient');
  });
});
runCheck('saved_stage_contracts',[],function(){
  var cfg={};Object.keys(CONFIG).forEach(function(k){cfg[k]=CONFIG[k];});
  cfg.assetRoot=PROBE_ASSETS.assetRoot;cfg.runId=PROBE_ASSETS.runId;cfg.fixture='small';
  cfg.step=PROBE_ASSETS.step;cfg.stage='final';validateConfig(cfg);
  assertProbe(cfg.assetRoot&&cfg.step>=2,'assets requires explicit root and saved step >=2');
  var paths=stageAssetPaths(cfg.assetRoot,cfg),metadata={};
  function token(path){if(!metadata[path])metadata[path]=ee.data.getAsset(path);var m=metadata[path];return JSON.stringify([path,m.updateTime||m.startTime||'',m.sizeBytes||'']);}
  var loaded=[];
  [[paths.components,'components',undefined],[paths.prepared,'prepared',0],[paths.state(1),'state',1],[paths.state(cfg.step),'state',cfg.step]].forEach(function(item){
    var image=ee.Image(item[0]),info=image.getInfo(),props=info.properties;
    var components=loaded.length?loaded[0].props:undefined;
    if(components)components.component_token=token(paths.components);
    props=restoreStageProperties(props,info.bands,cfg,item[1],item[0],item[2],components,item[1]==='state'?token(paths.prepared):undefined);
    assertProbe(props.source_token==='synthetic-small-'+VERSION,'Unexpected synthetic source');
    if(loaded.length)assertProbe(JSON.stringify(props.grid_transform)===JSON.stringify(loaded[0].props.grid_transform)&&props.grid_crs===loaded[0].props.grid_crs&&props.region_json===loaded[0].props.region_json,'Stage grid/region mismatch');
    loaded.push({image:image,props:props});
  });
  var props=loaded[0].props,gr=validatedGrid({crs:props.grid_crs,transform:props.grid_transform},'assets',true),region=ee.Geometry(JSON.parse(props.region_json));
  var solver=createGEESolver(loaded[0].image,gr,region,cfg,ee,loaded[1].image),state=loaded[3].image;
  var invalidState=state.neq(state).or(state.abs().gte(1e11)).reduce(ee.Reducer.max())
    .or(state.mask().reduce(ee.Reducer.min()).eq(0)).unmask(1).updateMask(loaded[0].image.select('support'));
  var observed=ee.Dictionary({rows:solver.diagnostics(state.select('S'),solver.muFor(state),state).toList(5),
    invalid:solver.grid.total(invalidState.rename('v'),ee.Reducer.max()).get('v')}).getInfo();
  nearValue(observed.invalid,0,0,'Masked/non-finite saved state');
  var rows=observed.rows;
  assertProbe(rows.length===4,'Saved small fixture requires four components');
  rows.forEach(function(f){STATE_FIELDS.forEach(function(k){assertProbe(typeof f.properties[k]==='number'&&isFinite(f.properties[k]),'Invalid state field '+k);});assertProbe(f.properties.bad===0&&typeof f.properties.status==='number'&&f.properties.status%1===0&&f.properties.status>=0&&f.properties.status<=5,'Invalid saved state');});
  print('Saved state diagnostics (no iteration or export)',rows);
  if(rows.some(function(f){return f.properties.status===1||f.properties.status===2;})){
    print('Contracts passed; solution still running. Continue stages in main entry. Final products NOT yet validated.');
  }else{
    var result=solver.finalize(state),g=solver.grid;
    nearValue(pixelCount(result.valid,g),415,0,'Saved fixture final support');
    nearValue(maxValue(result.depth.subtract(.5).abs(),g),0,1e-7,'Saved fixture depth');
    nearValue(maxValue(result.gradient.abs(),g),0,1e-10,'Saved fixture flat gradient');
    print('Final products from saved state passed; no tasks created.');
  }
});
print('CFDepth v3.2.1 selected-group summary (no asset tasks):',acceptanceResults);
var acceptanceTotals=summarizeAcceptance(acceptanceResults);
print('CFDepth group / totals / timing includes client, network and server wait:',PROBE_GROUP,acceptanceTotals);
if(acceptanceTotals.failed||acceptanceTotals.skipped)throw Error('CFDepth acceptance incomplete: '+
  acceptanceTotals.passed+' PASS, '+acceptanceTotals.failed+' FAIL, '+acceptanceTotals.skipped+' SKIP; inspect selected group.');
print('PASS GROUP '+PROBE_GROUP+'. Other groups, asset persistence and large-scene execution remain separate gates.');
