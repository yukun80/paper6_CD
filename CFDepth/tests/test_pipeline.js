'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const makeGrid=require('./gee_array_mock');
const source=fs.readFileSync(path.join(__dirname,'../CFDepth_0919.txt'),'utf8'),ctx={module:{exports:{}}};
vm.runInNewContext(source,ctx);
const grid={crs:'EPSG:4326',transform:[1/3600,0,110,0,-1/3600,30]},cfg={...ctx.CONFIG};
let count=0;function test(name,body){body();count++;console.log('PASS '+name);}
function maxima(image){const b=image.first();return Math.max(...b.v.filter((v,i)=>b.m[i]));}
function sum(image){const b=image.first();return b.v.reduce((s,v,i)=>s+(b.m[i]?v:0),0);}
function setup(){
  const a=makeGrid(12,8,-2,-2),{Img,ee,n}=a,g=ctx.createGEEGrid(grid,{},cfg,ee);
  const values=Array.from({length:n},(_,i)=>{const[x,y]=a.coord(i);return x>=0&&x<=2&&y>=0&&y<=3?1:x>=5&&x<=7&&y>=0&&y<=3?2:0;});
  const ids=Img.one(values,'cid'),support=ids.gt(0).rename('support');
  const dem=g.C(99).rename('dem'),D=ctx.packStage(ee.Image.cat([dem,ids,support,support.not().rename('dry'),support.rename('hard')]),'components');
  const B=ee.Image.cat([g.C(99.25).rename('lower'),g.C(100).rename('upper'),g.C(99.5).rename('mid'),g.C(.8).rename('weight')]).updateMask(support);
  const prepared=D.addBands(B),solver=ctx.createGEESolver(D,grid,{},cfg,ee,prepared);
  return {...a,g,values,ids,support,D,B,solver};
}
test('pixelIndices floors half centers including negatives without shifting the grid',()=>{
  const a=makeGrid(5,5,-2,-2),p=a.ee.Projection(grid.crs,grid.transform),indices=ctx.pixelIndices(p,a.ee);
  for(let i=0;i<a.n;i++){
    assert.equal(indices.bands.x.v[i],a.coord(i)[0]);assert.equal(indices.bands.y.v[i],a.coord(i)[1]);
  }
  // 整数坐标返回形式也兼容，不做固定减0.5。
  const base=a.ee.Image.pixelCoordinates();a.ee.Image.pixelCoordinates=()=>base.subtract(.5);
  assert.deepEqual(ctx.pixelIndices(p,a.ee).bands,indices.bands);
});
test('four colors cover every pixel once and separate all eight neighbors',()=>{
  const a=makeGrid(8,8,-4,-4),colors=ctx.fourColors(ctx.pixelIndices({},a.ee)).first().v;
  for(let i=0;i<a.n;i++){
    assert.ok(Number.isInteger(colors[i])&&colors[i]>=0&&colors[i]<=3);
    for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){
      const x=i%8+dx,y=Math.floor(i/8)+dy;
      if((dx||dy)&&x>=0&&x<8&&y>=0&&y<8)assert.notEqual(colors[i],colors[y*8+x]);
    }
  }
  const old=ctx.fourColors(a.ee.Image.pixelCoordinates()).first().v;
  assert.ok(old.every(v=>![0,1,2,3].includes(v)),'Regression must reproduce old no-update colors');
});
test('production sweep changes non-equilibrium WSE, lowers energy, preserves hard bound and other component',()=>{
  const a=setup(),g=a.g,S=g.C(99.5).updateMask(a.support).rename('S');
  const perturbed=S.add(ctx.fourColors(g.xy()).eq(0).and(a.ids.eq(1)).multiply(.2));
  const before=sum(a.solver.energyImages(perturbed,g.C(0)).select('primary'));
  const next=a.solver.sweep(perturbed,g.C(0),a.support);
  const after=sum(a.solver.energyImages(next,g.C(0)).select('primary'));
  assert.ok(maxima(next.subtract(perturbed).abs())>1e-6);assert.ok(after<before,{before,after});
  assert.equal(maxima(g.C(99.01).subtract(next).max(0)),0);
  assert.ok(maxima(next.subtract(S).abs().updateMask(a.ids.eq(2)))<1e-10);
  assert.deepEqual(next.first().m,S.first().m);
});
test('shared sweep is stationary at analytic solution and inactive components remain unchanged',()=>{
  const a=setup(),S=a.g.C(99.5).updateMask(a.support).rename('S');
  assert.ok(maxima(a.solver.sweep(S,a.g.C(0),a.support).subtract(S).abs())<1e-10);
  const shifted=S.add(.2);assert.deepEqual(a.solver.sweep(shifted,a.g.C(0),a.g.C(0)).bands,shifted.rename('S').bands);
});
test('real state pack contract excludes DEM, and old versions/snapshots cannot restore',()=>{
  assert.equal(ctx.VERSION,'cfdepth-soft-interval-v3.2.1');assert.equal(ctx.CONFIG.runId,'cfdepth_v321_run01');
  const a=setup(),images=Array.from(ctx.stageBandNames('prepared'),name=>a.g.C(0).rename(name));
  const image=a.ee.Image.cat(images),state=ctx.packStage(image,'state');
  assert.deepEqual(Object.keys(state.bands),['S','baseS',...Array.from(ctx.STATE_FIELDS)]);
  assert.throws(()=>ctx.packStage(state,'prepared'),/Missing selected band dem/);
  const props={version:ctx.VERSION,config_signature:ctx.configSignature(cfg),kind:'state',run_id:cfg.runId,step:1,component_token:'c',prepared_token:'p'};
  assert.doesNotThrow(()=>ctx.validateStageMetadata(props,cfg,'state',1,'c','p'));
  for(const changes of [{version:'cfdepth-soft-interval-v3.0'},{version:'cfdepth-soft-interval-v3.2'},{step:2},{component_token:'old'},{prepared_token:undefined},{run_id:'old'}]){
    assert.throws(()=>ctx.validateStageMetadata({...props,...changes},cfg,'state',1,'c','p'));
  }
  assert.throws(()=>ctx.validateStageMetadata(props,cfg,'nonsense'),/Unknown stage/);
});
test('shared gradient uses the correct neighbor direction and preserves isolated NoData',()=>{
  const a=setup(),g=a.g,x=g.xy().select('x'),y=g.xy().select('y');
  const plane=x.multiply(2).add(y.multiply(3)).toDouble().updateMask(a.support);
  const result=a.solver.gradientImages(plane,a.support),b=result.gradient.first();
  for(let i=0;i<a.n;i++)if(b.m[i]){
    const lat=(30-(a.coord(i)[1]+.5)/3600)*Math.PI/180,dx=6378137*Math.cos(lat)*Math.PI/180/3600,dy=6378137*Math.PI/180/3600;
    assert.ok(Math.abs(b.v[i]-Math.hypot(2/dx,3/dy))<1e-9);
  }
  const isolated=x.eq(1).and(y.eq(1)),single=a.solver.gradientImages(plane.updateMask(isolated),isolated);
  assert.ok(single.gradient.first().m.every(v=>v===0));
});
test('in-memory production sweep after exact state repack matches continuous sweep',()=>{
  const a=setup(),S=a.g.C(99.5).add(ctx.fourColors(a.g.xy()).eq(0).multiply(.2)).updateMask(a.support).rename('S');
  const first=a.solver.sweep(S,a.g.C(0),a.support),names=Array.from(ctx.stageBandNames('state'));
  const image=a.ee.Image.cat(names.map(name=>name==='S'?first:a.g.C(0).rename(name)));
  const restored=ctx.packStage(image,'state'),rebuilt=ctx.createGEESolver(a.D,grid,{},cfg,a.ee,a.D.addBands(a.B));
  const continuous=a.solver.sweep(first,a.g.C(0),a.support),restarted=rebuilt.sweep(restored.select('S'),a.g.C(0),a.support);
  assert.deepEqual(restarted.bands,continuous.bands);
});
test('acceptance runner collects failures and skips only dependent groups',()=>{
  const cases=fs.readFileSync(path.join(__dirname,'probes/acceptance_cases.js'),'utf8');
  const runner={print(){},PROBE_GROUP:'quick'};
  vm.runInNewContext(cases.slice(cases.indexOf('var acceptanceResults='),cases.indexOf('function maxValue(')),runner);
  runner.probeGroups.quick.push('failure','dependent','independent');
  runner.runCheck('failure',[],()=>{throw Error('Simulated server failure');});
  runner.runCheck('dependent',['failure'],()=>{throw Error('Must not execute');});
  let executed=false;runner.runCheck('independent',[],()=>{executed=true;});
  assert.equal(executed,true);
  assert.deepEqual(Array.from(runner.acceptanceResults,r=>r.status),['FAIL','SKIP','PASS']);
});
test('standalone acceptance includes exact production code and never creates tasks and has no unrolled iteration chain',()=>{
  const probe=fs.readFileSync(path.join(__dirname,'gee_acceptance_probe.js'),'utf8');
  assert.ok(probe.includes(source.slice(0,source.indexOf('function runGEE('))));
  assert.ok(!probe.includes('Export.image.'));assert.ok(!probe.includes('runGEE('));
  const cases=fs.readFileSync(path.join(__dirname,'probes/acceptance_cases.js'),'utf8');
  assert.ok(cases.includes('createGEESolver(')&&cases.includes('.sweep(')&&!cases.includes('.iterate(')&&cases.includes('.finalize('));
});
console.log(`${count} shared pipeline tests passed; array adapter is not a GEE cloud acceptance.`);
