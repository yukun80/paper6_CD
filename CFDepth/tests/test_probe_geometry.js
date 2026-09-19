'use strict';
// 验收脚本的几何接口/调度契约测试；不模拟真实投影精度。
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const cases=fs.readFileSync(path.join(__dirname,'probes/acceptance_cases.js'),'utf8');
const prelude=cases.slice(0,cases.indexOf("runCheck('coordinates_geographic'"));
const geographic=[1/3600,0,110,0,-1/3600,30],projected=[10,0,1918543,0,-10,7241630],rotated=[10,2,1918543,1,-10,7241630];
let count=0;function test(name,body){body();count++;console.log('PASS '+name);}
function harness(options={},code=prelude){
  const calls={transforms:[],margins:[],samples:[]};
  function margin(value,unit='meters'){
    if(unit==='meters'&&value!==0&&value<.001)throw Error('Invalid ErrorMargin value');
    return {value,unit};
  }
  const ee={Projection(crs,transform){return {crs,transform};},
    ErrorMargin(value,unit){calls.margins.push({value,unit});return margin(value,unit);},
    Geometry:{Polygon(){return {};}}};
  function image(projection){return {rename(){return this;},addBands(){return this;},sample(args){
    calls.samples.push(args);
    const t=projection.transform,features=[];
    for(let y=-2;y<2;y++)for(let x=-2;x<2;x++){
      const properties={column:x,row:y,raw_x:x+.5,raw_y:y+.5};
      const point=[t[0]*(x+.5)+t[1]*(y+.5)+t[2],t[3]*(x+.5)+t[4]*(y+.5)+t[5]];
      const feature={properties,set(key,value){properties[key]=value;return this;},geometry(){
        assert.equal(args.geometries,true,'Index-only sampling must not request geometries');
        return {transform(crs,maxError){
          const m=typeof maxError==='number'?margin(maxError):margin(maxError.value,maxError.unit);
          calls.transforms.push({crs,margin:m});
          if(options.failTransform)throw Error('Simulated point reprojection failure');
          return {coordinates(){return [point[0]+(options.halfPixelShift?t[0]*.5:0),point[1]];}};
        }};
      }};
      features.push(feature);
    }
    return {map(fn){features.forEach(fn);return this;},getInfo(){return {features};}};
  }};}
  ee.Image={pixelCoordinates:image};
  const ctx={CONFIG:{diagnosticDetails:false},PROBE_GROUP:options.group||'geometry',ee,pixelIndices:image,print(){},grid:{transform:geographic},nullCountProbe(){}};
  vm.runInNewContext(code,ctx);return {ctx,calls};
}
test('old micrometre error fails while explicit zero metre point transform succeeds',()=>{
  const old=harness({},prelude.replace("ee.ErrorMargin(0,'meters')",'0.000001'));
  assert.throws(()=>old.ctx.coordinateCenterProbe('EPSG:3857',projected),/Invalid ErrorMargin/);
  const {ctx,calls}=harness();ctx.coordinateCenterProbe('EPSG:3857',projected);
  assert.equal(calls.transforms.length,16);assert.equal(calls.margins.length,16);
  assert.ok(calls.margins.every(m=>m.value===0&&m.unit==='meters'));
  assert.throws(()=>ctx.ee.ErrorMargin(1e-6,'meters'),/Invalid ErrorMargin/);
});
test('index checks do not create geometries or invoke any transform',()=>{
  const {ctx,calls}=harness({failTransform:true});
  for(const [crs,t]of [['EPSG:4326',geographic],['EPSG:3857',projected],['EPSG:3857',rotated]])ctx.coordinateIndexProbe(crs,t);
  assert.equal(calls.transforms.length,0);assert.equal(calls.margins.length,0);
  assert.ok(calls.samples.every(s=>s.geometries===false&&s.dropNulls===false));
});
test('physical checks preserve tight tolerance and reject a half pixel displacement',()=>{
  for(const [crs,t]of [['EPSG:4326',geographic],['EPSG:3857',projected],['EPSG:3857',rotated]]){
    assert.doesNotThrow(()=>harness().ctx.coordinateCenterProbe(crs,t));
    assert.throws(()=>harness({halfPixelShift:true}).ctx.coordinateCenterProbe(crs,t),/Physical pixel center x/);
  }
});
test('each selected group has complete ordered dependencies and nonselected bodies never run',()=>{
  const {ctx}=harness({failTransform:true}),registered=[];
  ctx.runCheck=(name,deps,body)=>registered.push({name,deps,body});
  vm.runInNewContext(cases.slice(cases.indexOf("runCheck('coordinates_geographic'"),cases.indexOf("print('CFDepth v3.2.1 selected-group summary")),ctx);
  assert.equal(new Set(registered.map(r=>r.name)).size,registered.length);
  for(const group of Object.keys(ctx.probeGroups)){
    const run=harness({group}).ctx;
    for(const r of registered){
      if(run.probeGroups[group].includes(r.name))assert.ok(r.deps.every(d=>run.acceptanceStatus[d]==='PASS'),group+': missing dependency '+r.name);
      run.runCheck(r.name,r.deps,()=>assert.ok(run.probeGroups[group].includes(r.name),'Unselected body ran'));
    }
    assert.equal(run.summarizeAcceptance(run.acceptanceResults).failed,0);
    assert.equal(run.summarizeAcceptance(run.acceptanceResults).skipped,0);
    assert.equal(run.summarizeAcceptance(run.acceptanceResults).passed,run.probeGroups[group].length);
    assert.ok(run.acceptanceResults.every(r=>Number.isFinite(r.elapsed_ms)&&r.elapsed_ms>=0));
  }
  const run=harness({failTransform:true}).ctx;
  for(const r of registered)run.runCheck(r.name,r.deps,r.name.startsWith('coordinates_')||r.name.startsWith('physical_centers_')?r.body:()=>{});
  const totals=run.summarizeAcceptance(run.acceptanceResults);
  assert.equal(totals.failed,3);assert.equal(totals.skipped,0);assert.equal(totals.passed,6);
  assert.equal(run.acceptanceStatus.production_components,'NOT_RUN');
  assert.ok(!cases.includes('.iterate('),'Probe must not unroll the production iterative chain');
});
test('summary rejects selected failures/skips, but not unselected work',()=>{
  const {ctx}=harness();
  ctx.runCheck('coordinates_geographic',[],()=>{throw Error('Bad indices');});
  ctx.runCheck('physical_centers_geographic',['coordinates_geographic'],()=>assert.fail('must skip'));
  ctx.runCheck('production_components',['coordinates_geographic'],()=>assert.fail('not selected'));
  assert.deepEqual(Array.from(ctx.acceptanceResults,r=>r.status),['FAIL','SKIP','NOT_RUN']);
  const footer=cases.slice(cases.indexOf("print('CFDepth v3.2.1 selected-group summary"));
  assert.throws(()=>vm.runInNewContext(footer,ctx),/0 PASS, 1 FAIL, 1 SKIP/);
  ctx.acceptanceResults=[{status:'PASS'},{status:'NOT_RUN'}];
  assert.doesNotThrow(()=>vm.runInNewContext(footer,ctx));
  assert.equal(ctx.summarizeAcceptance(ctx.acceptanceResults).not_run,1);
  assert.throws(()=>harness({group:'unknown'}),/Unknown PROBE_GROUP/);
});
console.log(`${count} probe geometry/dependency tests passed; real GEE transforms are not simulated.`);
