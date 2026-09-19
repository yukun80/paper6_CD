'use strict';
// No Earth Engine credentials: execute the actual production scalar kernels.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const context={module:{exports:{}}};
vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../CFDepth_0919.txt'),'utf8'),context);
const api=context.module.exports, cfg=api.CONFIG, op=api.numericOps();
let count=0;
function test(name,fn){fn();count++;console.log('PASS '+name);}
function close(a,b,tol=1e-10){assert.ok(Math.abs(a-b)<=tol,`${a} != ${b}`);}
function coord(v){return api.coordinateMinimum(v,op);}
function point(overrides={}){return Object.assign({degree:2,neighborSum:10,boundary:10,lower:1,upper:2,
  soft:0,terrain:0,hard:false,midWeight:0,mid:1.5},overrides);}
function energy(v,s){return v.degree*s*s-2*v.neighborSum*s+v.boundary*Math.max(0,v.lower-s,s-v.upper)**2+
  v.soft*Math.max(0,v.terrain-s)**2+v.midWeight*(s-v.mid)**2;}
function derivative(v,s){return 2*(v.degree*s-v.neighborSum+v.boundary*(s<v.lower?s-v.lower:s>v.upper?s-v.upper:0)+
  v.soft*Math.min(0,s-v.terrain)+v.midWeight*(s-v.mid));}
function independentRoot(v){
  if(v.hard&&derivative(v,v.terrain)>=0)return v.terrain;
  let lo=v.hard?v.terrain:-10000,hi=10000;
  for(let i=0;i<100;i++){const m=(lo+hi)/2;if(derivative(v,m)>0)hi=m;else lo=m;}
  return (lo+hi)/2;
}
test('piecewise coordinate minimum matches independent monotone derivative root',()=>{
  let seed=42;function rand(){seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/2**32;}
  for(let i=0;i<1500;i++){
    let lo=rand()*40-20;
    const v=point({degree:0.1+rand()*8,neighborSum:rand()*100-50,boundary:rand()*20,
      lower:lo,upper:lo+rand()*10,terrain:rand()*30-15,soft:rand()*10,hard:rand()>0.5,
      midWeight:rand()*0.1,mid:lo+2});
    const got=coord(v),expected=independentRoot(v);close(got,expected,1e-9);
    assert.ok(energy(v,got)<=energy(v,expected)+1e-8);
    if(v.hard)assert.ok(got>=v.terrain);
  }
});
test('soft boundary can move; midpoint is not restored',()=>{
  const v=point();const result=coord(v);assert.ok(result>v.upper);assert.notEqual(result,v.mid);
  close(coord(point({boundary:0,soft:0})),5);
});
test('hard interior and soft edge give different terrain behavior',()=>{
  const v=point({degree:1,neighborSum:0,boundary:0,terrain:10,soft:1});
  close(coord(v),5);close(coord({...v,hard:true}),10);
});
function stat(median,mad=0,count=6){return {median,mad,count};}
test('boundary ordering, dispersion, slope, count and peer conflicts reduce weight',()=>{
  const call=(w,d,s=0,p=null)=>api.boundaryTerms(w,d,s,p,cfg,op);
  const good=call(stat(1),stat(2));assert.ok(good.weight>=0.5);close(good.lower,.75);close(good.upper,2.25);
  assert.ok(call(stat(1.2),stat(1)).weight<call(stat(1),stat(1.2)).weight);
  close(call(stat(5),stat(1)).weight,0); // Never swap an inverted interval.
  assert.ok(call(stat(1,2),stat(2,2)).weight<good.weight);
  assert.ok(call(stat(1),stat(2),15).weight<good.weight);
  assert.ok(call(stat(1,0,1),stat(2)).weight<good.weight);
  assert.ok(call(stat(1),stat(2),0,stat(10)).weight<good.weight);
  close(call(stat(1),stat(2),0,stat(10,0,2)).weight,good.weight);
  close(call(stat(1,0,0),stat(2)).weight,0);
});
test('component eligibility rejects missing, sparse and spatially concentrated anchors',()=>{
  const eligible=(n,x0,x1,y0,y1)=>api.hasBoundarySupport(n,x0,x1,y0,y1,cfg,op);
  assert.equal(eligible(0,1e12,-1e12,1e12,-1e12),false);
  assert.equal(eligible(2,0,100,0,100),false);
  assert.equal(eligible(3,0,1,0,1),false);
  assert.equal(eligible(3,0,2,0,0),true);
  assert.equal(eligible(3,0,0,0,2),true);
});
function state(overrides={}){return Object.assign({status:1,attempt:0,sweeps:0,stable:0,
  prev_primary:2,prev_total:2,base_primary:0,base_mid:0},overrides);}
function result(overrides={}){return Object.assign({primary:2,total:2,mid:3,residual:0,hard:0,bad:0},overrides);}
function advance(s,r,c=cfg){return api.advanceComponent(s,r,c,op);}
test('requires two stable stages AND coordinate stationarity',()=>{
  let s=advance(state(),result());assert.equal(s.status,1);assert.equal(s.stable,1);
  const stalled=advance(s,result({residual:.1}));assert.equal(stalled.status,1);
  s=advance(s,result());assert.equal(s.status,2);assert.equal(s.saveBase,true);
  close(s.base_primary,2);close(s.prev_total,2+cfg.lambdaB*cfg.muRatios[0]*3);
});
test('primary failure on iteration limit, invalid values, increasing objective, or hard violation',()=>{
  for(const r of [result({residual:1}),result({bad:1}),result({hard:1}),result({total:3})]){
    const s=advance(state({sweeps:cfg.maxSweeps-cfg.sweepsPerStage}),r);
    assert.equal(s.status,5);assert.equal(s.saveBase,false);
  }
});
test('secondary budget: accept / lower mu and restart base / final fallback',()=>{
  const mu=cfg.lambdaB*cfg.muRatios[0];
  const s=state({status:2,stable:1,base_primary:2,base_mid:3,prev_primary:2.0001,prev_total:2.0001+mu*3});
  assert.equal(advance(s,result({primary:2.0001,total:2.0001+mu*3})).status,3);
  const high=2.01, rejected=advance({...s,prev_primary:high,prev_total:high+mu*3},result({primary:high,total:high+mu*3}));
  assert.equal(rejected.status,2);assert.equal(rejected.attempt,1);assert.equal(rejected.restoreBase,true);
  close(rejected.prev_primary,2);close(rejected.prev_total,2+cfg.lambdaB*cfg.muRatios[1]*3);
  const last=advance({...s,attempt:cfg.muRatios.length-1,prev_primary:high,prev_total:high},result({primary:high,total:high}));
  assert.equal(last.status,4);assert.equal(last.restoreBase,true);
});
test('zero primary objective uses absolute budget; independent component decisions',()=>{
  const s=state({status:2,stable:1,base_primary:0,base_mid:1,prev_primary:1e-9,prev_total:.01});
  assert.equal(advance(s,result({primary:1e-9,total:.01})).status,3);
  assert.equal(advance({...s,prev_primary:1e-5},result({primary:1e-5,total:.01})).status,2);
});
test('serialized stage state gives the same transitions as continuous state',()=>{
  let a=state(),b=state();
  for(let i=0;i<8;i++){
    const r=result({primary:2,total:a.status===1?2:a.prev_total});
    a=advance(a,r);b=advance(JSON.parse(JSON.stringify(b)),r);
    assert.deepEqual(JSON.parse(JSON.stringify(a)),JSON.parse(JSON.stringify(b)));
  }
});
test('an accepted midpoint solution must retain its nonzero mu for final residual',()=>{
  const v=point({degree:1,neighborSum:0,boundary:0,midWeight:.01,mid:10});
  const solution=coord(v);close(coord(v)-solution,0);
  assert.ok(Math.abs(coord({...v,midWeight:0})-solution)>cfg.residualTolerance);
});
test('central/one-sided gradients and absent axes remain distinct',()=>{
  const d=(c,p,m,dp,dm,hp,hm)=>api.directionalDifference(c,p,m,dp,dm,hp,hm,op);
  close(d(10,12,8,20,20,true,true).value,.1);
  close(d(10,12,0,20,20,true,false).value,.1);
  close(d(10,0,8,20,20,false,true).value,.1);
  assert.equal(d(10,0,0,20,20,false,false).valid,false);
  close(d(7,7,7,20,20,true,true).value,0);
  close(d(10,13,8,30,20,true,true).value,.1);
  for(const lat of [0,30,60]){
    const dx=6378137*Math.cos(lat*Math.PI/180)*Math.PI/180/3600;
    const dy=6378137*Math.PI/180/3600;
    close(d(10,10+.02*dx,10-.02*dx,dx,dx,true,true).value,.02);
    close(d(10,10+.03*dy,0,dy,dy,true,false).value,.03);
  }
});

// Small independent grid driver: production minimizer, explicit edge-list energy.
function gridFixture(){
  const rows=6,cols=7,n=rows*cols;
  const labels=Array.from({length:n},(_,i)=>(i%cols<3?1:i%cols>3?2:0));
  const neighbors=Array.from({length:n},()=>[]),edges=[];
  for(let p=0;p<n;p++)if(labels[p])for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){
    if(!dx&&!dy)continue;let y=Math.floor(p/cols)+dy,x=p%cols+dx;
    if(y<0||y>=rows||x<0||x>=cols)continue;let q=y*cols+x;
    if(labels[q]!==labels[p])continue;const w=1/(dx*dx+dy*dy);
    neighbors[p].push([q,w]);if(q>p)edges.push([p,q,w]);
  }
  const v=Array.from({length:n},(_,p)=>point({degree:neighbors[p].reduce((a,b)=>a+b[1],0),neighborSum:0,
    boundary:(Math.floor(p/cols)===0||Math.floor(p/cols)===rows-1)?10:0,
    lower:labels[p]===1?2:12,upper:labels[p]===1?3:13,mid:labels[p]===1?2.5:12.5,
    terrain:labels[p]===1?1:11,soft:1,hard:false,midWeight:0}));
  v[0].lower=4;v[0].upper=4.5; // Conflicting soft boundary, not a fixed anchor.
  function F(S){let f=0;for(const [p,q,w]of edges)f+=w*(S[p]-S[q])**2;
    for(let p=0;p<n;p++)if(labels[p])f+=v[p].boundary*Math.max(0,v[p].lower-S[p],S[p]-v[p].upper)**2+
      v[p].soft*Math.max(0,v[p].terrain-S[p])**2;return f;}
  let S=labels.map(c=>c===1?2.5:c===2?12.5:0),history=[F(S)];
  for(let iter=0;iter<4000;iter++){
    for(let color=0;color<4;color++)for(let p=0;p<n;p++)if(labels[p]&&p%cols%2+2*(Math.floor(p/cols)%2)===color){
      S[p]=coord({...v[p],neighborSum:neighbors[p].reduce((a,[q,w])=>a+w*S[q],0)});
    }
    history.push(F(S));
  }
  return {S,history,labels,edges,v,rows,cols};
}
const fixture=gridFixture();
test('four-color sweeps decrease energy and disconnected component stays unchanged',()=>{
  for(let i=1;i<fixture.history.length;i++)assert.ok(fixture.history[i]<=fixture.history[i-1]+1e-10);
  fixture.labels.forEach((c,p)=>{if(c===2)close(fixture.S[p],12.5,1e-8);});
  assert.ok(fixture.S[0]<4); // Compromise is allowed at the conflicting anchor.
});
if(process.argv.includes('--dump-fixture')){
  const destination=process.argv[process.argv.indexOf('--dump-fixture')+1];
  if(!destination)throw new Error('--dump-fixture requires an output path');
  fs.writeFileSync(destination,JSON.stringify(fixture));
}
console.log(`${count} numerical tests passed; GEE server execution is NOT covered.`);
