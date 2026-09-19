'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const source=fs.readFileSync(path.join(__dirname,'../CFDepth_0919.txt'),'utf8');
const context={module:{exports:{}}};vm.runInNewContext(source,context);
const cfg=context.CONFIG,dem={crs:'EPSG:4326',transform:[1/3600,0,110,0,-1/3600,30]};
const projected={crs:'EPSG:3857',transform:[9.32480563590904,0,1918543.3940000013,0,-9.324805635909165,7241630.245534113]};
let count=0;
function test(name,fn){fn();count++;console.log('PASS '+name);}
const WKT='PARAM_MT["Affine",PARAMETER["num_row",3],PARAMETER["num_col",3]]';
function mockRaster(grid,stats){
  let args,transformCalls=0;
  const region={tag:'original footprint'};
  const image={projection(){return {getInfo(){return grid;},transform(){transformCalls++;return WKT;}};},
    geometry(){return region;},neq(){return this;},and(){return this;},
    reduceRegion(value){args=value;return {getInfo(){return stats;}};}};
  const countReducer={name:'count'};
  const ee={Reducer:{count(){return countReducer;},max(){return {combine(options){
    assert.equal(options.reducer2,countReducer);assert.equal(options.sharedInputs,true);return {name:'max_count'};
  }};}}};
  return {image,ee,get args(){return args;},get transformCalls(){return transformCalls;},region};
}
test('actual input reduction uses numeric affine, original geometry and resolution',()=>{
  for(const grid of [dem,projected,{crs:'EPSG:3857',transform:[10,2,100,1,-10,200]}]){
    const mock=mockRaster(grid,{raw_max:0,raw_count:15});
    const result=context.validateInputRaster(mock.image,cfg,mock.ee);
    assert.equal(result.validPixels,15);assert.equal(mock.transformCalls,0);
    assert.deepEqual(Array.from(mock.args.crsTransform),grid.transform);
    assert.equal(mock.args.crs,grid.crs);assert.equal(mock.args.geometry,mock.region);
    assert.equal(mock.args.bestEffort,false);assert.ok(!('scale' in mock.args));
  }
});
test('malformed or singular grids fail before raster reduction',()=>{
  for(const grid of [null,{}, {crs:'',transform:dem.transform}, {crs:dem.crs,transform:WKT},
    {crs:dem.crs,transform:[1,0,0,0,-1]}, {crs:dem.crs,transform:[1,0,0,0,0,0]},
    {crs:dem.crs,transform:[1,2,0,2,4,0]}, {crs:dem.crs,transform:[1,0,Infinity,0,-1,0]},
    {crs:dem.crs,transform:[1,0,0,0,NaN,0]}, {crs:dem.crs,transform:['1',0,0,0,-1,0]}]){
    const mock=mockRaster(grid,{raw_max:0,raw_count:1});
    assert.throws(()=>context.validateInputRaster(mock.image,cfg,mock.ee),/components \/ input flood raster/);
    assert.equal(mock.args,undefined);
  }
});
test('input grids allow projected and rotated coordinates while DEM checks stay strict',()=>{
  assert.doesNotThrow(()=>context.validatedGrid(projected,'input',false));
  assert.throws(()=>context.validatedGrid(projected,'DEM',true),/FABDEM requires/);
  assert.throws(()=>context.validatedGrid({...dem,transform:[1,0.1,0,0,-1,0]},'DEM',true),/FABDEM requires/);
  assert.doesNotThrow(()=>context.validatedGrid(dem,'DEM',true));
  const args=context.gridArguments(dem,'export',true);args.crsTransform[0]=99;
  assert.equal(dem.transform[0],1/3600); // 校验不能改写原网格。
});
test('empty, unknown encoding and anomalous reductions have distinct errors',()=>{
  for(const n of [0,null])assert.throws(()=>context.validateInputStats({raw_count:n,raw_max:null},'input'),/no valid pixels/);
  assert.throws(()=>context.validateInputStats({raw_count:15,raw_max:1},'input'),/other than 0\/1/);
  for(const stats of [null,{}, {raw_count:'15',raw_max:0},{raw_count:NaN,raw_max:0},
    {raw_count:15,raw_max:null},{raw_count:15,raw_max:2}]){
    assert.throws(()=>context.validateInputStats(stats,'input'),/invalid/);
  }
});
function stage(kind){
  const available=['dem','cid','support','dry','hard','lower','upper','mid','weight','eligible','S','baseS',...context.STATE_FIELDS];
  const image={select(names){
    assert.ok(names.every(n=>available.includes(n)));
    return {toDouble(){return Array.from(names,id=>({id,crs:dem.crs,crs_transform:dem.transform.slice()}));}};
  }};
  return context.packStage(image,kind);
}
test('restored stage requires all bands and identical validated DEM grids',()=>{
  const props={grid_crs:dem.crs,grid_transform:dem.transform};
  assert.equal(stage('state')[0].id,'S');
  assert.ok(!stage('state').some(b=>b.id==='dem'));
  assert.throws(()=>stage('unknown'),/Unknown stage asset kind/);
  for(const kind of ['components','prepared','state']){
    const bands=stage(kind);assert.doesNotThrow(()=>context.validateStageBands(props,bands,kind,'restore'));
    assert.throws(()=>context.validateStageBands(props,bands.slice(1),kind,'restore'),/missing band/);
    bands[0].crs_transform[2]+=1;
    assert.throws(()=>context.validateStageBands(props,bands,kind,'restore'),/differs from metadata/);
    bands[0].crs_transform=WKT;
    assert.throws(()=>context.validateStageBands(props,bands,kind,'restore'),/not WKT/);
  }
});
test('component grouped reducer arity and duplicate fields are checked',()=>{
  assert.doesNotThrow(()=>context.validateGrouping(['weight','xmin'],['sum','min'],'grouped'));
  for(const args of [[[],[]],[['x'],[]],[['x','x'],['sum','sum']],[['cid'],['sum']],[['x'],['median']]]){
    assert.throws(()=>context.validateGrouping(...args,'grouped'),/grouped/);
  }
});
// 独立 GEE 验收文件使用同一份网格/输入检查函数，避免测试副本与生产入口漂移。
test('standalone GEE projection probe contains the exact production validation helpers',()=>{
  const start=source.indexOf('function validatedGrid('),end=source.indexOf('function validateStageBands(');
  const probe=fs.readFileSync(path.join(__dirname,'gee_acceptance_probe.js'),'utf8');
  assert.ok(probe.includes(source.slice(start,end)));
});
console.log(`${count} projection/interface tests passed; GEE server execution is NOT covered.`);
