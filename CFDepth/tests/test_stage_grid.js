'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../CFDepth_0919.txt'),'utf8');
const c={module:{exports:{}}};vm.runInNewContext(source,c);
const grid={crs:'EPSG:4326',transform:[1/3600,0,110,0,-1/3600,30]};
const cfg={...c.CONFIG,fixture:'small',runId:'cfdepth_v321_small'};
const clone=x=>JSON.parse(JSON.stringify(x));
function asset(kind='components',format='json',g=grid){
  const props={version:c.VERSION,kind,config_signature:c.configSignature(cfg),run_id:cfg.runId,
    source_token:'synthetic-small-'+c.VERSION,grid_crs:g.crs,region_json:'small-region',
    component_token:'components-token',prepared_token:'prepared-token',step:kind==='prepared'?0:1};
  if(format==='json')Object.assign(props,c.stageGridProperties(g));
  if(format==='array')props.grid_transform=g.transform.slice();
  const bands=Array.from(c.stageBandNames(kind),id=>({id,crs:g.crs,crs_transform:g.transform.slice()}));
  return {props,bands};
}
function restore(a,components){return c.restoreStageProperties(a.props,a.bands,cfg,a.props.kind,'test restore',
  a.props.kind==='components'?undefined:a.props.kind==='prepared'?0:1,components,a.props.kind==='state'?'prepared-token':undefined);}
let n=0;function test(name,fn){fn();n++;console.log('PASS '+name);}
test('new string metadata survives array-dropping persistence and all stage restores',()=>{
  let components;
  for(const kind of ['components','prepared','state']){
    const a=asset(kind);a.props.grid_transform=grid.transform;
    // Model a storage service that omits array-valued custom properties.
    a.props=JSON.parse(JSON.stringify(a.props,(_key,value)=>Array.isArray(value)?undefined:value));
    const result=restore(a,components);assert.deepEqual(clone(result.grid_transform),grid.transform);
    assert.equal(typeof a.props.grid_transform_json,'string');assert.ok(!('grid_transform' in a.props));
    if(kind==='components')components=result;
  }
  assert.match(source,/var gridProps=stageGridProperties\(\{crs:CRS,transform:T\}\)/);
  assert.match(source,/crs:CRS,crsTransform:T,maxPixels/);
});
test('old arrays and matching dual properties are supported',()=>{
  const a=asset('components','array');restore(a);
  a.props.grid_transform_json=JSON.stringify(grid.transform);restore(a);
});
test('user small missing-property components restore without mutating raw metadata',()=>{
  const a=asset('components','missing'),before=clone(a);const result=restore(a);
  assert.deepEqual(clone(result.grid_transform),grid.transform);assert.deepEqual(a,before);
  for(const kind of ['prepared','state'])restore(asset(kind,'missing'),result);
});
test('conflicting properties and malformed present properties never fall back',()=>{
  const a=asset();a.props.grid_transform=grid.transform.slice();a.props.grid_transform[2]+=1;
  assert.throws(()=>restore(a),/conflicting/);
  for(const value of ['{','null','[1,2]',JSON.stringify([0,0,110,0,0,30]),null,{},123]){
    const b=asset();b.props.grid_transform_json=value;assert.throws(()=>restore(b),/JSON|transform|singular|invertible/);
  }
  for(const value of [null,undefined,'WKT',[Infinity,0,110,0,-1/3600,30]]){
    const b=asset('components','array');b.props.grid_transform=value;assert.throws(()=>restore(b));
  }
});
test('missing bands and inconsistent actual grids fail for every format and kind',()=>{
  for(const format of ['json','array','missing'])for(const kind of ['components','prepared','state']){
    const a=asset(kind,format);a.bands.pop();assert.throws(()=>restore(a),/missing band/);
    const b=asset(kind,format);b.bands[1].crs_transform[2]+=1;
    assert.throws(()=>restore(b),/actual grid differs/);
  }
  const a=asset('components','missing');a.bands=[];assert.throws(()=>restore(a),/missing band/);
});
test('declared CRS and fixed small fixture grid are mandatory even without metadata',()=>{
  const a=asset('components','missing');a.props.grid_crs='EPSG:3857';assert.throws(()=>restore(a),/grid_crs/);
  for(const format of ['json','array','missing']){
    const b=asset('components',format,{...grid,transform:[1/3600,0,111,0,-1/3600,30]});
    assert.throws(()=>restore(b),/small fixture grid mismatch/);
  }
});
test('version, signature, source and snapshot validation cannot be bypassed by fallback',()=>{
  for(const key of ['version','config_signature','run_id','source_token']){
    const a=asset('components','missing');a.props[key]='wrong';assert.throws(()=>restore(a));
  }
  const components=restore(asset());
  for(const key of ['component_token','prepared_token','region_json']){
    const a=asset('state','missing');a.props[key]='wrong';assert.throws(()=>restore(a,components));
  }
  const a=asset('state');a.props.step=2;assert.throws(()=>restore(a,components),/sequence/);
});
test('cross-stage grid and source checks also apply to non-fixture assets',()=>{
  const realCfg={...cfg,fixture:''};
  const base=restore(asset());
  for(const kind of ['prepared','state']){
    const a=asset(kind);a.props.config_signature=c.configSignature(realCfg);
    a.props.grid_transform_json=JSON.stringify([1/3600,0,111,0,-1/3600,30]);
    a.bands.forEach(b=>b.crs_transform[2]=111);
    assert.throws(()=>c.restoreStageProperties(a.props,a.bands,realCfg,kind,'real',undefined,base,'prepared-token'),/stage grid differs/);
    const b=asset(kind);b.props.config_signature=c.configSignature(realCfg);b.props.source_token='different';
    assert.throws(()=>c.restoreStageProperties(b.props,b.bands,realCfg,kind,'real',undefined,base,'prepared-token'),/stage source differs/);
  }
});
console.log(n+' stage grid tests passed');
