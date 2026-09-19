'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../CFDepth_0919.txt'),'utf8');
const c={module:{exports:{}}};vm.runInNewContext(source,c);
const clone=x=>JSON.parse(JSON.stringify(x));
const opts={mode:'auto',pollSeconds:30,resume:true};
function job(key,destination='asset'){return {key,destination,description:'auto_'+key,path:key,binding:'fixed',stage:key};}
function harness(jobs,shared){
  const state=shared||{record:null,tasks:{},assets:{},starts:[],built:[],logs:[],clock:0};
  let queue=[],id=0;const delays=[];
  const io={owner:'page',identity:'identity',now:()=>state.clock,log:x=>state.logs.push(x),
    schedule(fn,ms){const key=++id;queue.push({fn,ms,key});delays.push(ms);return key;},cancel(key){queue=queue.filter(x=>x.key!==key);},
    read:()=>clone(state.record),save:r=>{state.record=clone(r);},create(r){assert.equal(state.record,null);state.record=clone(r);},
    checkIdentity:r=>assert.equal(r.identity,'identity'),checkFresh(resume){if(!resume)assert.equal(Object.keys(state.assets).length,0);},
    next(exports){for(const j of jobs){if(j.destination==='asset'?!state.assets[j.key]:!exports[j.key])return clone(j);}return null;},
    tasks:()=>Object.values(state.tasks).map(clone),newId:()=>`task-${Object.keys(state.tasks).length+1}`,
    build(j){state.built.push(j.key);return {description:j.description,key:j.key};},
    start(id,p,cb){assert.equal(state.record.current.attempted,true);assert.equal(state.record.current.id,id);
      assert.ok(!state.tasks[id]);state.starts.push(p.key);state.tasks[id]={id,description:p.description,state:'READY',key:p.key};cb({taskId:id});},
    status(id,cb){const t=state.tasks[id];cb(t?clone(t):{state:'UNKNOWN'});},
    verify:j=>!!state.assets[j.key]};
  const controller=c.createAutoController(io,opts);
  return {state,io,controller,delays,
    one(){assert.ok(queue.length,'expected queued callback');const q=queue.shift();state.clock+=q.ms;q.fn();},
    until(pred,limit=100){for(let n=0;!pred();n++){assert.ok(n<limit,JSON.stringify(state.logs));this.one();}},
    complete(){const j=state.record.current;assert.ok(j);state.tasks[j.id].state='COMPLETED';if(j.destination==='asset')state.assets[j.key]=true;},
    get pending(){return queue.length;}};
}
let n=0;function test(name,fn){fn();n++;console.log('PASS '+name);}
test('fresh full chain waits for completion and both sequential exports',()=>{
  const jobs=[job('components'),job('prepare'),job('iterate1'),job('iterate2'),job('depth','drive'),job('gradient','drive')];
  const h=harness(jobs);h.controller.start();
  jobs.forEach((j,index)=>{
    h.until(()=>h.state.starts.length===index+1);assert.equal(h.state.starts[index],j.key);
    h.one();assert.equal(h.state.starts.length,index+1);assert.ok(!h.state.record.complete);
    h.complete();
  });
  h.until(()=>h.controller.isStopped());assert.equal(h.state.record.complete,true);assert.equal(h.state.record.leaseUntil,0);
  assert.deepEqual(Object.keys(h.state.record.exports),['depth','gradient']);
});
test('resume from completed iterate2 skips all prior computation',()=>{
  const jobs=[job('components'),job('prepare'),job('iterate1'),job('iterate2'),job('iterate3'),job('depth','drive'),job('gradient','drive')];
  const h=harness(jobs);jobs.slice(0,4).forEach(j=>h.state.assets[j.key]=true);h.controller.start();
  h.until(()=>h.state.starts.length);assert.deepEqual(h.state.starts,['iterate3']);
});
test('adopts READY and RUNNING tasks without rebuilding or submitting',()=>{
  for(const state of ['READY','RUNNING']){
    const h=harness([job('components')]);h.state.tasks.old={id:'old',description:'auto_components',state,key:'components'};
    h.controller.start();h.until(()=>h.state.record.current);h.one();assert.deepEqual(h.state.starts,[]);assert.deepEqual(h.state.built,[]);
    h.complete();h.until(()=>h.controller.isStopped());assert.equal(h.state.record.complete,true);
  }
});
test('COMPLETED task waits for asset visibility and validates before advancing',()=>{
  const h=harness([job('components'),job('prepare')]);h.controller.start();h.until(()=>h.state.starts.length);
  h.state.tasks[h.state.record.current.id].state='COMPLETED';h.one();h.one();assert.deepEqual(h.state.starts,['components']);
  h.state.assets.components=true;h.until(()=>h.state.starts.length===2);
});
test('visibility timeout fails without starting dependent task',()=>{
  const h=harness([job('components'),job('prepare')]);h.controller.start();h.until(()=>h.state.starts.length);
  h.state.tasks[h.state.record.current.id].state='COMPLETED';h.until(()=>h.controller.isStopped());
  assert.deepEqual(h.state.starts,['components']);assert.match(h.state.logs.at(-1),/unavailable/);
});
test('FAILED CANCELLED and CANCEL_REQUESTED stop without resubmission',()=>{
  for(const status of ['FAILED','CANCELLED','CANCEL_REQUESTED']){
    const h=harness([job('components'),job('prepare')]);h.controller.start();h.until(()=>h.state.starts.length);
    Object.assign(h.state.tasks[h.state.record.current.id],{state:status,error_message:'original reason'});
    h.until(()=>h.controller.isStopped());assert.match(h.state.logs.at(-1),/original reason/);assert.equal(h.state.starts.length,1);
  }
});
test('prior failed and ambiguous histories are rejected',()=>{
  const h=harness([job('components')]);h.state.tasks.old={id:'old',description:'auto_components',state:'FAILED'};
  h.controller.start();h.until(()=>h.controller.isStopped());assert.equal(h.state.starts.length,0);
  assert.throws(()=>c.autoTaskMatch([{description:'x'},{description:'x'}],{description:'x'}),/Ambiguous/);
  assert.equal(c.autoTaskMatch([{}],{description:'x'}),null);
});
test('query retries reissue the request after 30/60/120 seconds then stop',()=>{
  const h=harness([job('components')]);let calls=0;h.io.status=(_id,cb)=>{calls++;cb(null,'503 network unavailable');};
  h.controller.start();h.until(()=>h.state.starts.length);h.until(()=>h.controller.isStopped());
  assert.equal(calls,4);assert.deepEqual(h.delays.slice(-3),[30000,60000,120000]);assert.equal(h.state.starts.length,1);
});
test('transient query recovery does not rebuild submitted expression',()=>{
  const h=harness([job('components')]);let calls=0;const original=h.io.status;
  h.io.status=(id,cb)=>++calls<3?cb(null,'network timeout'):original(id,cb);
  h.controller.start();h.until(()=>h.state.starts.length);h.complete();h.until(()=>h.controller.isStopped());
  assert.equal(calls,3);assert.equal(h.state.built.length,1);assert.equal(h.state.record.complete,true);
});
test('lost submission response polls saved request ID and never resends',()=>{
  const h=harness([job('components')]),start=h.io.start;
  h.io.start=(id,p,cb)=>start(id,p,()=>cb(null,'network response lost'));
  h.controller.start();h.until(()=>h.state.starts.length);h.one();h.complete();h.until(()=>h.controller.isStopped());
  assert.equal(h.state.starts.length,1);
});
test('unknown submission cannot be interpreted as permission to retry',()=>{
  const h=harness([job('components')]);h.io.start=(_id,_p,cb)=>cb(null,'response lost');
  h.controller.start();h.until(()=>h.controller.isStopped());assert.match(h.state.logs.at(-1),/refusing duplicate/);
  assert.equal(h.state.built.length,1);
});
test('page restart preserves current task and completed final export receipts',()=>{
  const jobs=[job('depth','drive'),job('gradient','drive')];const h=harness(jobs);h.controller.start();
  h.until(()=>h.state.starts.length);h.complete();h.until(()=>h.state.starts.length===2);h.controller.stop();
  h.state.clock+=300001;const resumed=harness(jobs,h.state);resumed.io.owner='other';
  // owner is captured at controller creation; use the shared page ID after lease expiry.
  resumed.controller.start();resumed.one();resumed.complete();resumed.until(()=>resumed.controller.isStopped());
  assert.deepEqual(resumed.state.starts,['depth','gradient']);assert.equal(resumed.state.record.complete,true);
});
test('competing owner cannot submit while a lease is active or stolen',()=>{
  const h=harness([job('components')]);h.controller.start();h.state.record.owner='another';h.one();
  assert.equal(h.controller.isStopped(),true);assert.equal(h.state.starts.length,0);
  const second=harness([job('components')]);second.state.record={schema:'cfdepth-auto-1',identity:'identity',owner:'other',leaseUntil:999999,exports:{}};
  second.controller.start();assert.equal(second.state.starts.length,0);assert.match(second.state.logs.at(-1),/owns this run/);
});
test('contract or final audit failure prevents submission',()=>{
  const h=harness([job('depth','drive')]);h.io.build=()=>{throw Error('Final residual audit failed');};h.controller.start();
  h.until(()=>h.controller.isStopped());assert.equal(h.state.starts.length,0);assert.match(h.state.logs.at(-1),/audit failed/);
});
test('stage inventories require contiguous bounded sequence',()=>{
  const cfg=c.CONFIG,root='projects/p/assets',pre=root+'/'+cfg.runId;
  assert.equal(c.autoStageLimit(cfg),1000);
  const base=[pre+'_components',pre+'_prepared',pre+'_state_00001',pre+'_state_00002'];
  assert.deepEqual(clone(c.autoInventory(base,root,cfg)).steps,[1,2]);
  for(const ids of [[pre+'_prepared'],[pre+'_state_00001'],[...base,pre+'_state_00004'],[...base,pre+'_state_01001'],[...base,pre+'_state_bad']])
    assert.throws(()=>c.autoInventory(ids,root,cfg));
});
test('export adapter preserves element affine pyramiding NoData and Drive filenames',()=>{
  const args={image:{image:true},description:'task',region:{geometry:true},crs:'EPSG:4326',crsTransform:[1,0,2,0,-1,3],
    maxPixels:1e13,assetId:'projects/p/assets/x',pyramidingPolicy:{'.default':'sample'},folder:'FloodDepth',fileNamePrefix:'CFDepth',formatOptions:{noData:-9999}};
  const a=c.autoExportParams('asset',args),d=c.autoExportParams('drive',args);
  assert.equal(a.element,args.image);assert.equal(a.crs_transform,args.crsTransform);assert.equal(a.overwrite,false);
  assert.equal(a.pyramidingPolicy['.default'],'SAMPLE');assert.equal(args.pyramidingPolicy['.default'],'sample');assert.equal(d.tiffNoData,-9999);assert.equal(d.fileFormat,'GEO_TIFF');
  assert.equal(d.driveFolder,'FloodDepth');assert.equal(d.driveFileNamePrefix,'CFDepth');assert.ok(!('assetId' in d));
});
test('run options do not enter numerical configuration and defaults remain compatible',()=>{
  assert.equal(c.RUN_OPTIONS.mode,'auto');assert.equal(c.VERSION,'cfdepth-soft-interval-v3.2.1');
  assert.equal(c.autoDigest('x'),c.autoDigest('x'));assert.notEqual(c.autoDigest('x'),c.autoDigest('y'));
  for(const changes of [{mode:'other'},{pollSeconds:0},{resume:'yes'}])assert.throws(()=>c.validateRunOptions({...opts,...changes}));
  assert.equal(Object.keys(c.CONFIG).includes('mode'),false);
});

test('production GEE adapter resumes saved assets and emits both correctly named Drive tasks',()=>{
  const env={module:{exports:{}}};vm.runInNewContext(source,env);
  const cfg={...env.CONFIG,fixture:'small',assetRoot:'projects/p/assets',runId:'small-auto'},grid={crs:'EPSG:4326',transform:[1/3600,0,110,0,-1/3600,30]};
  const prefix=cfg.assetRoot+'/'+cfg.runId,store={},tasks={},queue=[],logs=[],writes=[],builds=[];
  const token=id=>JSON.stringify([id,'v1',10]);
  function add(suffix,kind,step){const id=prefix+suffix;
    store[id]={type:'Image',updateTime:'v1',sizeBytes:10,properties:{version:env.VERSION,kind,step,run_id:cfg.runId,
      config_signature:env.configSignature(cfg),source_token:'synthetic-small-'+env.VERSION,
      grid_crs:grid.crs,grid_transform_json:JSON.stringify(grid.transform),region_json:'{}',
      component_token:token(prefix+'_components'),prepared_token:token(prefix+'_prepared')},
      bands:Array.from(env.stageBandNames(kind),id=>({id,crs:grid.crs,crs_transform:grid.transform}))};}
  add('_components','components');add('_prepared','prepared',0);add('_state_00001','state',1);add('_state_00002','state',2);
  const originalInputs=clone(store);let clock=0,seq=0;
  const data={getAsset(id){if(!store[id])throw Error('not found');return clone(store[id]);},
    listAssets(root){assert.equal(root,cfg.assetRoot);return {assets:Object.keys(store).map(id=>({id}))};},
    listOperations:()=>Object.entries(tasks).map(([id,t])=>({name:id,metadata:{description:t.description,state:t.state==='COMPLETED'?'SUCCEEDED':t.state}})),
    createAsset(value,id,force,properties){if(value.type==='Folder'&&properties)throw Error('Extraneous field(s) present: [properties].');assert.equal(value.type,'ImageCollection');assert.equal(properties,undefined);assert.equal(force,false);assert.ok(!store[id]);store[id]={type:'ImageCollection',properties:{}};},
    setAssetProperties(id,p){writes.push(id);store[id].properties=clone(p);},newTaskId:()=>['id-'+(++seq)],
    startProcessing(id,p,cb){assert.equal(p.type,'EXPORT_IMAGE');assert.equal(p.tiffNoData,-9999);assert.equal(p.crs_transform.length,6);
      assert.ok(!tasks[id.replace(/-/g,'')]);tasks[id.replace(/-/g,'')]={state:'COMPLETED',description:p.description,file:p.driveFileNamePrefix};cb({taskId:id.replace(/-/g,'')});},
    getTaskStatus(id,cb){assert.ok(!id.includes('-'));cb([tasks[id]||{state:'UNKNOWN'}]);}};
  const image=id=>({getInfo:()=>clone(store[id]),select(){return this;},rename(){return this;},addBands(){return this;}});
  const table={map(){return this;},aggregate_histogram:()=>({'3':2,'0':2}),size:()=>4,filter:()=>({size:()=>0})};
  env.ee={data,Image:image,Geometry:x=>x,Dictionary:x=>({getInfo:()=>x}),Filter:{eq:()=>({})}};
  env.createGEESolver=()=>({grouped:()=>table});
  env.print=(...args)=>logs.push(args);env.ui={util:{setTimeout(fn,ms){queue.push({fn,ms});return queue.length;},clearTimeout(){}}};
  env.Date={now:()=>clock};
  env.runGEE=(config,input,runtime)=>{
    builds.push(config.stage);assert.equal(config.stage,'final');assert.equal(config.step,2);
    for(const suffix of ['','_WSE_Gradient'])runtime.emit('drive',{image:{},description:cfg.exportName+suffix,
      crs:grid.crs,crsTransform:grid.transform,region:{},maxPixels:1e13,folder:cfg.exportFolder,
      fileNamePrefix:cfg.exportName+suffix,formatOptions:{noData:-9999}});
  };
  const controller=env.runAutoGEE(cfg,null,opts);
  for(let i=0;!controller.isStopped();i++){assert.ok(i<40,JSON.stringify(logs));const q=queue.shift();assert.ok(q);clock+=q.ms;q.fn();}
  assert.deepEqual(Object.values(tasks).map(t=>t.file),['CFDepth_v3','CFDepth_v3_WSE_Gradient']);
  assert.deepEqual(builds,['final']);assert.ok(writes.every(id=>id===prefix+'_automation_record'));
  for(const id of Object.keys(originalInputs))assert.deepEqual(clone(store[id]),originalInputs[id]);
  // 再次Run只读收据，不再构图或导出。
  const again=env.runAutoGEE(cfg,null,opts);
  for(let i=0;!again.isStopped();i++){assert.ok(i<10);const q=queue.shift();clock+=q.ms;q.fn();}
  assert.equal(Object.keys(tasks).length,2);assert.equal(builds.length,1);
});
test('receipt write response loss does not lose current job or duplicate final export',()=>{
  const h=harness([job('depth','drive'),job('gradient','drive')]);let failed=false;const save=h.io.save;
  h.io.save=r=>{save(r);if(r.exports.depth&&!r.current&&!failed){failed=true;throw Error('network response lost');}};
  h.controller.start();h.until(()=>h.state.starts.length===1);h.complete();h.until(()=>h.state.starts.length===2);
  h.complete();h.until(()=>h.controller.isStopped());assert.deepEqual(h.state.starts,['depth','gradient']);assert.equal(failed,true);
});
console.log(n+' total automation tests passed; includes adapter integration with simulated GEE.');
