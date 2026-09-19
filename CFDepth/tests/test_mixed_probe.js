'use strict';
// 仅验证GEE返回值的逐像元/多字段断言，不伪造云端执行结果。
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const source=fs.readFileSync(path.join(__dirname,'probes/acceptance_cases.js'),'utf8');
const checks=source.slice(source.indexOf('  function equal(a,b,message)'),source.indexOf("  print('PASS fractional masks"));
const names=['a','peak','b','floor'],definitions=[{key:'aligned',groups:[1,2]},{key:'non_aligned',groups:[1]}];
function fixture(){
  const results={};
  for(const def of definitions){
    const aligned=def.key==='aligned',rows=aligned?[{cid:1,a:2.6,b:13.2,peak:17,floor:-8},{cid:2,a:6,b:24,peak:27,floor:-6}]:[{cid:1,a:11.28627450980392,b:20,peak:37,floor:-8}];
    const reference=Object.fromEntries(rows.map(r=>[r.cid,{...r}])),samples=[];
    for(let y=0;y<2;y++)for(let x=0;x<(aligned?4:3);x++){
      const cid=aligned?(x<2?1:x===2?2:0):1;
      const before=aligned&&x===0&&y===1?0:x===1?.4:1;
      const a={column:x,row:y,source_cid:cid,cid_mask:cid>0?1:0,fraction_control:.4};
      names.forEach(n=>{a[n+'_before']=before;a[n+'_after']=cid>0?before:0;});samples.push({properties:a});
    }
    results[def.key]={grouped:rows.map(properties=>({properties})),reference,samples};
    if(!aligned)results[def.key].boundary_samples=[
      {properties:{column:3,row:0,peak:37,original_mask:.2,filtered_mask:.2}},
      {properties:{column:4,row:0,peak:0,original_mask:0,filtered_mask:0}}];
  }return results;
}
let count=0;function test(name,body){body();count++;console.log('PASS '+name);}
function verify(results){vm.runInNewContext(checks,{names,definitions,results,errors:[]});}
test('aligned and original nonaligned references pass all field and mask assertions',()=>{
  assert.doesNotThrow(()=>verify(fixture()));
  assert.ok(source.indexOf("print('Mixed reducers: all fields")<source.indexOf('  function equal(a,b,message)'));
});
test('failed sums and mask changes are reported together instead of stopping at field a',()=>{
  const r=fixture();r.non_aligned.grouped[0].properties.a=13.184313725490195;
  r.non_aligned.grouped[0].properties.b+=2;r.aligned.samples[1].properties.a_after=1;
  assert.throws(()=>verify(r),e=>/field=a/.test(e.message)&&/field=b/.test(e.message)&&/mask 1,0 a/.test(e.message));
});
test('missing/duplicate components and masked-background admission fail',()=>{
  for(const edit of [r=>r.aligned.grouped.pop(),r=>r.aligned.grouped.push(r.aligned.grouped[0]),r=>r.aligned.samples[3].properties.b_after=1]){
    const r=fixture();edit(r);assert.throws(()=>verify(r),/missing cid|duplicate cid|mask/);
  }
});
test('reference bounds preserve sum geometry and include overlapping extrema pixels',()=>{
  const helper={};vm.runInNewContext(source.slice(source.indexOf('function reducerReferenceBounds('),source.indexOf('function mixedReducerProbe(')),helper);
  const bounds=[.2,.2,3.2,2.2];
  assert.deepEqual(Array.from(helper.reducerReferenceBounds(bounds,'sum')),bounds);
  for(const method of ['min','max'])assert.deepEqual(Array.from(helper.reducerReferenceBounds(bounds,method)),[0,0,4,3]);
  assert.deepEqual(Array.from(helper.reducerReferenceBounds([-1.2,-.2,2.8,3.1],'max')),[-2,-1,3,4]);
  assert.deepEqual(bounds,[.2,.2,3.2,2.2]);
  // 独立枚举说明27/37来自不同入选范围；不能靠修改数值断言忽略边界。
  const cols=[0,1,2,3,4],center=cols.filter(x=>x+.5>=bounds[0]&&x+.5<bounds[2]);
  const overlap=cols.filter(x=>Math.min(x+1,bounds[2])>Math.max(x,bounds[0]));
  assert.equal(Math.max(...center.map(x=>10*x+7)),27);
  assert.equal(Math.max(...overlap.map(x=>10*x+7)),37);
});
test('old 27 reference fails even if grouped value is also 27; outside pixels remain excluded',()=>{
  const old=fixture();old.non_aligned.reference[1].peak=27;
  old.non_aligned.grouped[0].properties.peak=27;
  assert.throws(()=>verify(old),/Reference must include intersecting boundary/);
  for(const edit of [r=>r.non_aligned.boundary_samples[0].properties.original_mask=0,
    r=>r.non_aligned.boundary_samples[1].properties.original_mask=1,
    r=>r.non_aligned.boundary_samples.pop()]){
    const r=fixture();edit(r);assert.throws(()=>verify(r),/boundary|Exterior/);
  }
});
console.log(`${count} mixed-probe assertion tests passed; no GEE server execution.`);
