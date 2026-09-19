'use strict';
// 明确模拟GEE加权输入顺序；像元权重用于字段映射测试，不能代替真实云端归约。
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const makeGrid=require('./gee_array_mock'),oldMinimum=require('./fixtures/coordinate_minimum_v31');
const source=fs.readFileSync(path.join(__dirname,'../CFDepth_0919.txt'),'utf8'),ctx={module:{exports:{}}};
vm.runInNewContext(source,ctx);
let count=0;function test(name,fn){fn();count++;console.log('PASS '+name);}
class Reducer{
  constructor(fields){this.fields=fields;}
  setOutputs(names){return new Reducer(this.fields.map((f,i)=>({...f,name:names[i]})));}
  combine({reducer2,sharedInputs}){
    assert.equal(sharedInputs,false);
    if(this.fields.some(f=>!f.weighted)&&reducer2.fields.some(f=>f.weighted))throw Error('unweighted before weighted');
    return new Reducer([...this.fields,...reducer2.fields]);
  }
  group(opts){this.groupOptions=opts;return this;}
}
function reducers(){return Object.fromEntries(['sum','min','max'].map(kind=>[kind,()=>new Reducer([{kind,weighted:kind==='sum'}]) ]));}
test('old diagnostics ordering fails; stable layout puts all sums before extrema',()=>{
  const ee={Reducer:reducers()},names=['primary','residual','bad','status'],methods=['sum','max','sum','min'];
  assert.throws(()=>ctx.buildGroupedReducer({names,methods},ee),/unweighted before weighted/);
  const layout=ctx.groupedLayout(names,methods,'test'),r=ctx.buildGroupedReducer(layout,ee);
  assert.deepEqual(Array.from(layout.names),['primary','bad','residual','status']);
  assert.deepEqual(r.fields.map(f=>f.weighted),[true,true,false,false]);
  assert.equal(r.groupOptions.groupField,4);assert.deepEqual(methods,['sum','max','sum','min']);
});
test('actual grouped() maps differently valued/masked fields in sorted order with original sum weights',()=>{
  const a=makeGrid(4,2),{ee,Img,n}=a,grid={crs:'EPSG:4326',transform:[1/3600,0,110,0,-1/3600,30]};
  ee.Reducer=reducers();ee.List=x=>x;ee.Dictionary=x=>x;ee.Feature=(unused,properties)=>({properties});ee.FeatureCollection=x=>x;
  // 区域覆盖率与中心规则分开；边界/掩膜不能偷偷把加权sum改成整数计数。
  const coverage=[.8,1,.2,0,.8,1,.2,0],center=[1,1,0,0,1,1,0,0];
  let actualOrder;
  Img.prototype.reduceRegion=function(args){
    actualOrder=Object.keys(this.bands);const fields=args.reducer.fields,label=Object.values(this.bands)[fields.length];
    assert.equal(actualOrder[fields.length],'cid');const groups={};
    for(let i=0;i<n;i++){
      if(!label.m[i]||!label.v[i])continue;
      const group=groups[label.v[i]]||(groups[label.v[i]]={cid:label.v[i]});
      fields.forEach((f,j)=>{
        const band=Object.values(this.bands)[j],w=f.weighted?Math.min(band.m[i],coverage[i]):(band.m[i]>0&&center[i]?1:0);
        if(!w)return;
        if(f.kind==='sum')group[f.name]=(group[f.name]||0)+w*band.v[i];
        else group[f.name]=group[f.name]===undefined?band.v[i]:Math[f.kind](group[f.name],band.v[i]);
      });
    }
    return {get(){return Object.values(groups);}};
  };
  const ids=Img.one([1,1,1,0,2,2,2,0],'cid'),support=ids.gt(0).rename('support');
  const D=ee.Image.cat([Img.one(Array(n).fill(99),'dem'),ids,support,support.not().rename('dry'),support.rename('hard')]);
  const image=ee.Image.cat([
    Img.one([1,2,3,4,5,6,7,8],'a',[1,.4,1,1,1,0,1,1]),
    Img.one([70,90,800,4,30,10,600,8],'peak'),
    Img.one([6,9,12,15,18,21,24,27],'b',[1,.7,1,1,1,1,1,1]),
    Img.one([-8,-7,-6,-5,-4,-3,-2,-1],'floor')]);
  const rows=ctx.createGEESolver(D,grid,{},ctx.CONFIG,ee).grouped(image,['a','peak','b','floor'],['sum','max','sum','min']);
  assert.deepEqual(actualOrder,['a','b','peak','floor','cid']);
  const expected=[
    {cid:1,a:2.2,b:13.5,peak:90,floor:-8},
    {cid:2,a:5.4,b:40.2,peak:30,floor:-4}];
  assert.equal(rows.length,2);rows.forEach((r,i)=>Object.keys(expected[i]).forEach(k=>assert.ok(Math.abs(r.properties[k]-expected[i][k])<1e-12,k)));
});
test('updateMask adopts new fractional values without unmasking zeros; multiband masks stay separate',()=>{
  const a=makeGrid(4,1),{Img}=a;
  const input=Img.one([2,3,4,5],'a',[.4,0,1,1]).addBands(Img.one([6,7,8,9],'b',[1,.4,0,1]));
  const before=JSON.stringify(input.bands),all=Img.one([1,1,1,1]);
  assert.deepEqual(input.updateMask(all).bands.a.m,[1,0,1,1]);
  const mask=Img.one([.2,1,.4,0],'ma').addBands(Img.one([.4,1,1,0],'mb'));
  assert.deepEqual(input.updateMask(mask).bands.a.m,[.2,0,.4,0]);
  assert.deepEqual(input.updateMask(mask).bands.b.m,[.4,1,0,0]);
  const hidden=Img.one([1,1,1,1],'mask',[1,0,1,1]);
  assert.deepEqual(input.updateMask(hidden).bands.b.m,[1,0,0,1]);
  assert.equal(JSON.stringify(input.bands),before);
  assert.throws(()=>input.updateMask(mask.addBands(all.rename('extra'))),/arity/);
});
test('production grouped input preserves per-band fractions and rejects invalid component membership',()=>{
  const a=makeGrid(9,1),{Img,ee}=a;
  const ids=Img.one([1,1,2,0,-1,1.5,2**53,3,NaN],'cid',[.2,1,1,1,1,1,1,0,1]);
  const input=Img.one(Array(9).fill(7),'a',[.4,0,1,1,.4,.4,1,1,1])
    .addBands(Img.one(Array(9).fill(11),'b',[1,.4,.2,1,1,1,1,1,1]));
  const before=JSON.stringify([input.bands,ids.bands]);
  const filtered=ctx.groupedInput(input,['a','b'],ids,ee,{});
  assert.deepEqual(filtered.bands.a.m,[.4,0,1,0,0,0,0,0,0]);
  assert.deepEqual(filtered.bands.b.m,[1,.4,.2,0,0,0,0,0,0]);
  assert.deepEqual(filtered.bands.cid.m,[1,1,1,0,0,0,0,0,0]);
  assert.equal(JSON.stringify([input.bands,ids.bands]),before);
  const old=input.addBands(ids).updateMask(ids.gt(0));
  assert.equal(old.bands.a.m[0],1,'Old code must reproduce fractional weight loss');
  assert.notDeepEqual(old.bands.a.m,filtered.bands.a.m);
});
test('cached six-region template is bit-identical to v3.1 across 2000 varied constraints',()=>{
  let seed=123;const random=()=>((seed=(1664525*seed+1013904223)>>>0)/4294967296),o=ctx.numericOps();
  for(let i=0;i<2000;i++){
    const v={degree:i%17===0?0:random()*10,neighborSum:random()*500,boundary:random()*20,
      lower:random()*100,soft:i%3?random()*3:0,terrain:random()*110,hard:i%2===0,midWeight:i%5?random()*.1:0,mid:random()*100};
    v.upper=v.lower+random()*10;
    const template=ctx.coordinateTemplate(v,o);assert.equal(template.regions.length,6);
    assert.equal(ctx.coordinateMinimumFromTemplate(v.neighborSum,template,o),oldMinimum(v,o));
    for(const delta of [-1e-12,0,1e-12])assert.equal(ctx.coordinateMinimumFromTemplate(v.neighborSum+delta,template,o),oldMinimum({...v,neighborSum:v.neighborSum+delta},o));
  }
});
test('static shift cache keys source identity and offset and never crosses grid instances',()=>{
  let calls=0;const at=(image,x,y)=>({image,x,y,n:++calls}),cache=ctx.createStaticShiftCache(at),A={},B={};
  assert.equal(cache(A,1,-1),cache(A,1,-1));assert.equal(calls,1);
  assert.notEqual(cache(A,1,-1),cache(B,1,-1));assert.notEqual(cache(A,1,-1),cache(A,-1,1));
  assert.notEqual(cache(A,1,-1),ctx.createStaticShiftCache(at)(A,1,-1));assert.equal(calls,4);
  const a=makeGrid(3,3),grid={crs:'EPSG:4326',transform:[1/3600,0,110,0,-1/3600,30]},g=ctx.createGEEGrid(grid,{},ctx.CONFIG,a.ee);
  assert.equal(g.xy(),g.xy());assert.equal(g.colors(),g.colors());assert.equal(g.metricDistance(1,0),g.metricDistance(1,0));
  assert.notEqual(g.metricDistance(1,0),g.metricDistance(0,1));
});
test('minimum template is reused for a stage mu but rebuilt on changed mu',()=>{
  const a=makeGrid(3,3),grid={crs:'EPSG:4326',transform:[1/3600,0,110,0,-1/3600,30]},g=ctx.createGEEGrid(grid,{},ctx.CONFIG,a.ee);
  const D=g.pack([g.C(99).rename('dem'),g.C(1).rename('cid'),g.C(1).rename('support'),g.C(0).rename('dry'),g.C(1).rename('hard')]);
  const B=g.pack([g.C(99.25).rename('lower'),g.C(100).rename('upper'),g.C(99.8).rename('mid'),g.C(.8).rename('weight')]);
  const original=ctx.coordinateTemplate;let builds=0;ctx.coordinateTemplate=(...args)=>{builds++;return original(...args);};
  try{
    const solver=ctx.createGEESolver(D,grid,{},ctx.CONFIG,a.ee,D.addBands(B)),S=g.C(99.5),mu=g.C(0);
    solver.sweep(S,mu,g.C(1));assert.equal(builds,1);
    solver.minimum(S,mu);assert.equal(builds,1);
    const changed=solver.minimum(S,g.C(.01));assert.equal(builds,2);
    assert.ok(changed.first().v.some(x=>x>99.5));
    solver.minimum(S,mu);assert.equal(builds,3);
  }finally{ctx.coordinateTemplate=original;}
});
test('diagnostic detail switch is nonnumerical and default remains off',()=>{
  assert.equal(ctx.CONFIG.diagnosticDetails,false);
  assert.equal(ctx.configSignature(ctx.CONFIG),ctx.configSignature({...ctx.CONFIG,diagnosticDetails:true}));
  assert.equal(ctx.CONFIG.sweepsPerStage,10);assert.equal(ctx.CONFIG.maxSweeps,2000);
  assert.throws(()=>ctx.validateConfig({...ctx.CONFIG,diagnosticDetails:'false'}),/diagnosticDetails/);
});
console.log(`${count} optimization tests passed; weighted reducer adapter is not a GEE cloud result.`);
