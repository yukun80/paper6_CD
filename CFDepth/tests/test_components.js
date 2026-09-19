'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const source=fs.readFileSync(path.join(__dirname,'../CFDepth_0919.txt'),'utf8');
const ctx={module:{exports:{}}};vm.runInNewContext(source,ctx);
const grid={crs:'EPSG:4326',transform:[1/3600,0,110,0,-1/3600,30]};
let count=0;function test(name,fn){fn();count++;console.log('PASS '+name);}
// 小型逐像元运算适配器：仅验证生产检查表达式及归约权重；不是GEE矢量化模拟。
function harness(n,includeNull=true){
  class Img {
    constructor(values,mask=Array(n).fill(1),name='constant'){this.bands={[name]:{v:values,m:mask}};}
    first(){return Object.values(this.bands)[0];}
    reproject(){return this;}toByte(){return this;}
    remap(from,to,defaultValue){const a=this.first();return new Img(a.v.map(v=>{const i=from.indexOf(v);return i<0?defaultValue:to[i];}),a.m.slice());}
    rename(name){const b=this.first();return new Img(b.v.slice(),b.m.slice(),name);}
    binary(other,fn){const a=this.first(),b=other instanceof Img?other.first():{v:Array(n).fill(other),m:Array(n).fill(1)};
      return new Img(a.v.map((v,i)=>+fn(v,b.v[i])),a.m.map((v,i)=>Math.min(v,b.m[i])));}
    gt(b){return this.binary(b,(a,b)=>a>b);}lte(b){return this.binary(b,(a,b)=>a<=b);}
    gte(b){return this.binary(b,(a,b)=>a>=b);}neq(b){return this.binary(b,(a,b)=>a!==b);}
    mod(b){return this.binary(b,(a,b)=>a%b);}and(b){return this.binary(b,(a,b)=>!!a&&!!b);}
    or(b){return this.binary(b,(a,b)=>!!a||!!b);}not(){return this.binary(0,a=>!a);}
    unmask(value){const a=this.first();return new Img(a.v.map((v,i)=>a.m[i]?v:value),a.m.map(v=>v||1));}
    where(test,value){const a=this.first(),t=test.first();return new Img(a.v.map((v,i)=>t.m[i]&&t.v[i]?value:v),a.m.slice());}
    updateMask(mask){const a=this.first(),m=mask.first();return new Img(a.v.slice(),a.m.map((v,i)=>v>0&&m.m[i]>0?m.v[i]:0),Object.keys(this.bands)[0]);}
    reduceRegion(args){assert.deepEqual(Array.from(args.crsTransform),grid.transform);assert.equal(args.bestEffort,false);
      const result={};for(const [name,a]of Object.entries(this.bands)){
        if(args.reducer.kind==='distinct'){
          result[name]=new Set(a.v.map((v,i)=>a.m[i]>0?v:null)).size;
        }else if(args.reducer.kind==='distinct_nonnull'){
          result[name]=new Set(a.v.filter((v,i)=>a.m[i]>0&&v!==null)).size;
        }else if(args.reducer.kind==='histogram'){
          const histogram={};a.v.forEach((v,i)=>{
            const key=a.m[i]>0&&v!==null?v:'null';
            if(key!=='null'||includeNull)histogram[key]=(histogram[key]||0)+1;
          });
          result[name]=histogram;
        }else result[name]=a.v.reduce((sum,v,i)=>sum+(a.m[i]>0?v*(args.reducer.unw?1:a.m[i]):0),0);

      }
      return {getInfo(){return result;},get(key){return {getInfo(){return result[key];}};}};
    }
  }
  const reducer=kind=>({kind,unw:false,unweighted(){return {...this,unw:true};}});
  const ee={Image:{constant(v){return new Img(Array(n).fill(v));},cat(images){const out=new Img([]);out.bands={};
    images.forEach(im=>Object.assign(out.bands,im.bands));return out;}},Projection(crs,transform){return {crs,transform};},
    Dictionary(obj){function plain(v){if(v&&typeof v.getInfo==='function')return plain(v.getInfo());if(v&&typeof v==='object')return Object.fromEntries(Object.entries(v).map(([k,x])=>[k,plain(x)]));return v;}return {getInfo(){return plain(obj);}};},
    Reducer:{sum(){return reducer('sum');},countDistinct(){return reducer('distinct');},countDistinctNonNull(){return reducer('distinct_nonnull');},frequencyHistogram(){return reducer('histogram');}}};
  return {Img,ee};
}
function audit(a,b,weights,vectorIds,includeNull=true){
  const {Img,ee}=harness(a.length,includeNull);
  if(vectorIds===undefined)vectorIds=Array.from(new Set(b.filter(v=>v>0&&Number.isSafeInteger(v))));
  const result=ctx.auditComponentRaster(new Img(a,undefined,'support'),new Img(b,weights,'cid'),
    new Img(b,undefined,'cid'),grid,{},{...ctx.CONFIG,diagnosticDetails:true},ee,vectorIds);
  result.counts.vector_count=vectorIds.length;return result;
}
test('default compact diagnostics omit legacy reducers but retain exact safety checks',()=>{
  const {Img,ee}=harness(3);ee.Reducer.countDistinct=()=>{throw Error('Legacy distinct should be disabled');};
  const good=ctx.auditComponentRaster(new Img([1,0,1]),new Img([7,0,9]),new Img([7,0,9]),grid,{},ctx.CONFIG,ee,[7,9]);
  good.counts.vector_count=2;
  assert.equal(ctx.componentAuditError(good.counts),null);
  assert.equal(good.counts.weighted_expected,undefined);assert.equal(good.counts.distinct_ids_including_null,undefined);
  const bad=ctx.auditComponentRaster(new Img([1,0,1]),new Img([7,0,8]),new Img([7,0,8]),grid,{},ctx.CONFIG,ee,[7,9]);
  bad.counts.vector_count=2;assert.match(ctx.componentAuditError(bad.counts),/IDs|histogram/);
  assert.equal(bad.counts.unknown_id_pixels,1);
});
test('equal totals with exchanged pixels are rejected by exact membership checks',()=>{
  const d=audit([1,0,1,0],[0,1,2,0]).counts;
  assert.equal(d.expected,d.restored);assert.equal(d.missing,1);assert.equal(d.added,1);
  assert.match(ctx.componentAuditError(d),/pixel membership/);
});
test('weighted mismatch alone does not reject identical membership',()=>{
  const d=audit([1,0,1,0],[1,0,2,0],[0.3,1,0.5,1]).counts;
  assert.notEqual(d.weighted_expected,d.weighted_restored);
  assert.equal(d.expected,2);assert.equal(d.restored,2);assert.equal(d.missing,0);assert.equal(d.added,0);
  assert.equal(ctx.componentAuditError(d),null);
});
test('masked restored pixels are missing, even when their underlying IDs are positive',()=>{
  const d=audit([1,0,1,0],[1,0,2,0],[0,1,1,1]).counts;
  assert.equal(d.missing,1);assert.match(ctx.componentAuditError(d),/pixel membership/);
});
test('fractional or out of exact range IDs and merged component counts fail',()=>{
  const fractional=audit([1,1],[1,1.5]).counts;
  assert.equal(fractional.invalid_ids,1);assert.match(ctx.componentAuditError(fractional),/component IDs/);
  const large=audit([1],[2**53]).counts;assert.equal(large.invalid_ids,1);
  assert.match(ctx.componentAuditError(large),/component IDs/);
  const d=audit([1,1],[1,1]).counts;d.vector_count=2;assert.match(ctx.componentAuditError(d),/component IDs/);
  d.missing=.1;assert.match(ctx.componentAuditError(d),/Invalid integer/);
});
test('empty support is rejected; audit does not change source arrays',()=>{
  assert.match(ctx.componentAuditError(audit([0,0],[0,0]).counts),/Empty flood support/);
  const a=[1,0,1],b=[1,0,2],before=JSON.stringify([a,b]);audit(a,b);
  assert.equal(JSON.stringify([a,b]),before);
});
test('five valid labels plus masked background counts six including null but five non-null',()=>{
  const d=audit([1,1,1,1,1,0],[210,220,230,240,502049,0],[1,1,1,1,1,0]).counts;
  assert.equal(d.distinct_ids_including_null,6);assert.equal(d.distinct_ids,5);
  assert.equal(d.vector_count,5);assert.equal(d.histogram_distinct_ids,5);
  assert.equal(d.missing_ids.length,0);assert.equal(d.unknown_ids.length,0);
  assert.equal(ctx.componentAuditError(d),null);
});
test('single and all-masked cases do not use a fixed subtraction',()=>{
  const single=audit([1,1],[210,210]).counts;
  assert.equal(single.distinct_ids_including_null,1);assert.equal(single.distinct_ids,1);
  assert.equal(ctx.componentAuditError(single),null);
  const empty=audit([0,0],[0,0],[0,0],[]).counts;
  assert.equal(empty.distinct_ids_including_null,1);assert.equal(empty.distinct_ids,0);
  assert.match(ctx.componentAuditError(empty),/Empty flood support/);
});
test('same-size different ID sets fail and unknown pixels join the diagnostic mask',()=>{
  const result=audit([1,1,1,0],[210,999,999,0],undefined,[210,220]);
  const d=result.counts;
  assert.equal(d.distinct_ids,d.vector_count);
  assert.equal(d.missing_ids[0].id,220);assert.equal(d.missing_ids[0].pixels,0);
  assert.equal(d.unknown_ids[0].id,999);assert.equal(d.unknown_ids[0].pixels,2);
  assert.equal(d.unknown_id_pixels,2);assert.match(ctx.componentAuditError(d),/ID sets/);
  assert.deepEqual(Array.from(result.unknown.first().v),[0,1,1,0]);
  assert.deepEqual(Array.from(result.difference.first().v),[0,1,1,0]);
});
test('real extra and missing IDs fail; malformed histograms cannot pass',()=>{
  const extra=audit([1,1,1],[210,220,999],undefined,[210,220]).counts;
  assert.equal(extra.unknown_id_pixels,1);assert.match(ctx.componentAuditError(extra),/component IDs/);
  const missing=audit([1,1],[210,210],undefined,[210,220]).counts;
  assert.equal(missing.missing_ids[0].id,220);assert.match(ctx.componentAuditError(missing),/component IDs/);
  for(const histogram of [{undefined:1},{NULL:1},{0:1},{210:0},{210:.5},{210:'2'},{null:0},{null:-1},{null:.5},{null:'12'},{null:NaN},{null:Infinity},{null:2**53}]){
    assert.throws(()=>ctx.compareComponentIds([210],histogram),/Invalid component histogram/);
  }
});
test('histogram null buckets are diagnostic only; both backend formats pass',()=>{
  const a=[1,1,1,1,1,0],b=[210,220,230,240,502049,0];
  const withNull=audit(a,b).counts,without=audit(a,b,undefined,undefined,false).counts;
  assert.equal(withNull.histogram_null_count,1);assert.equal(without.histogram_null_count,0);
  assert.equal(withNull.raw_component_histogram.null,1);
  assert.equal(withNull.raster_id_pixels.null,undefined);
  assert.equal(withNull.histogram_pixels,5);
  assert.deepEqual(withNull.raster_id_pixels,without.raster_id_pixels);
  assert.equal(ctx.componentAuditError(withNull),null);assert.equal(ctx.componentAuditError(without),null);
});
test('null-only and empty histograms cannot hide five missing valid IDs',()=>{
  const labels=[210,220,230,240,502049],good=audit([1,1,1,1,1],labels).counts;
  for(const histogram of [{null:12},{}]){
    const sets=ctx.compareComponentIds(labels,histogram);
    assert.equal(sets.histogram_distinct_ids,0);assert.equal(sets.histogram_pixels,0);
    assert.equal(sets.missing_ids.length,5);
    assert.match(ctx.componentAuditError({...good,...sets}),/ID sets/);
  }
  assert.equal(ctx.compareComponentIds([], {null:12}).histogram_null_count,12);
});
test('null bucket is not subtracted from valid counts or used as a missing ID',()=>{
  const histogram={210:1,220:1,230:1,240:1,502049:1,null:12},before=JSON.stringify(histogram);
  const sets=ctx.compareComponentIds([210,220,230,240,502049],histogram);
  assert.equal(sets.histogram_pixels,5);assert.equal(sets.histogram_null_count,12);
  assert.equal(sets.histogram_distinct_ids,5);assert.equal(sets.missing_ids.length,0);
  assert.equal(JSON.stringify(histogram),before);
  const unknown=ctx.compareComponentIds([210,220],{210:1,999:1,null:12});
  assert.equal(unknown.unknown_ids[0].id,999);assert.equal(unknown.missing_ids[0].id,220);
});
test('independent 6x2 probe checks all positions, masks and exact labels',()=>{
  const probe=fs.readFileSync(path.join(__dirname,'gee_acceptance_probe.js'),'utf8'),checker={};
  vm.runInNewContext(probe.slice(probe.indexOf('function validateNullProbeSamples('),probe.indexOf('function nullCountProbe(')),checker);
  const labels=[210,220,230,240,502049];
  const features=Array.from({length:12},(_,i)=>{
    const x=i%6,y=Math.floor(i/6),wet=y===0&&x<5;
    return {properties:{column:x,row:y,support:+wet,support_mask:1,painted_mask:1,
      painted_cid:wet?labels[x]:0,cid:wet?labels[x]:null,cid_mask:+wet}};
  });
  assert.equal(checker.validateNullProbeSamples(features,labels),null);
  for(const field of ['support','support_mask','painted_mask','painted_cid','cid','cid_mask']){
    const bad=structuredClone(features);bad[0].properties[field]=null;
    assert.notEqual(checker.validateNullProbeSamples(bad,labels),null,field);
  }
  const shifted=structuredClone(features);shifted[0].properties.column=-1;
  assert.match(checker.validateNullProbeSamples(shifted,labels),/coordinate/);
  const duplicate=structuredClone(features);duplicate[1]=duplicate[0];
  assert.match(checker.validateNullProbeSamples(duplicate,labels),/coordinate/);
  assert.match(checker.validateNullProbeSamples(features.slice(1),labels),/12 sampled/);
  const allMasked=structuredClone(features);allMasked.forEach(f=>{f.properties.cid=null;f.properties.cid_mask=0;});
  assert.match(checker.validateNullProbeSamples(allMasked,labels),/valid ID/);
});
test('GEE probe uses exact production component builder and audit helpers',()=>{
  const probe=fs.readFileSync(path.join(__dirname,'gee_acceptance_probe.js'),'utf8');
  const start=source.indexOf('function componentIndicator('),end=source.indexOf('function runGEE(');
  assert.ok(probe.includes(source.slice(start,end)));
});
console.log(`${count} component audit tests passed; GEE rasterization is NOT simulated.`);
