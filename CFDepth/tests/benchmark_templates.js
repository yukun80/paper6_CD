'use strict';
// 只计数学表达式构造调用，不以本地运算时间推断GEE服务端加速。
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const old=require('./fixtures/coordinate_minimum_v31');
const ctx={module:{exports:{}}};vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../CFDepth_0919.txt'),'utf8'),ctx);
const v={degree:6,neighborSum:597,boundary:8,lower:99.25,upper:100,soft:1,terrain:99.01,hard:false,midWeight:.008,mid:99.5};
function counter(){let calls=0;const raw=ctx.numericOps(),ops={};Object.keys(raw).forEach(k=>{ops[k]=(...a)=>{calls++;return raw[k](...a);};});return {ops,count:()=>calls};}
const baseline=counter(),cached=counter(),template=ctx.coordinateTemplate(v,cached.ops);
for(let i=0;i<40;i++){
  const neighborSum=v.neighborSum+i*.01;
  const a=old({...v,neighborSum},baseline.ops),b=ctx.coordinateMinimumFromTemplate(neighborSum,template,cached.ops);
  if(a!==b)throw Error('Template numerical drift');
}
console.log(JSON.stringify({scope:'40 coordinate minimum evaluations, one fixed mu; numericOps construction calls only',baseline_calls:baseline.count(),template_calls:cached.count(),reduction_fraction:1-cached.count()/baseline.count(),cloud_timing_measured:false},null,2));
