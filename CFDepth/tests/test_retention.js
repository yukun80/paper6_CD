'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../CFDepth_0919.txt'),'utf8');
const clone=x=>JSON.parse(JSON.stringify(x));
function simulation({saved=2,cleanup=true,stopAt=4,legacy=false}={}){
  const env={module:{exports:{}}};vm.runInNewContext(source,env);
  const cfg={...env.CONFIG,fixture:'small',assetRoot:'projects/p/assets',runId:'retention-small'},grid={crs:'EPSG:4326',transform:[1/3600,0,110,0,-1/3600,30]};
  const pre=cfg.assetRoot+'/'+cfg.runId,store={},tasks={},queue=[],logs=[],deleted=[],writes=[],builds=[];
  let clock=0,sequence=0,timerId=0,controller;
  const token=id=>JSON.stringify([id,store[id].updateTime||'v1',10]);
  const hooks={};
  function add(step,kind='state'){
    const suffix=kind==='components'?'_components':kind==='prepared'?'_prepared':'_state_'+String(step).padStart(5,'0'),id=pre+suffix;
    store[id]={type:'Image',updateTime:'v1',sizeBytes:10,properties:{version:env.VERSION,kind,step,run_id:cfg.runId,
      config_signature:env.configSignature(cfg),source_token:'synthetic-small-'+env.VERSION,
      grid_crs:grid.crs,grid_transform_json:JSON.stringify(grid.transform),region_json:'{}',
      component_token:kind==='components'?'unused':token(pre+'_components'),prepared_token:kind==='state'?token(pre+'_prepared'):'unused'},
      bands:Array.from(env.stageBandNames(kind),id=>({id,crs:grid.crs,crs_transform:grid.transform}))};
    return id;
  }
  if(saved>=0){add(undefined,'components');add(0,'prepared');for(let step=1;step<=saved;step++)add(step);}
  const identity=JSON.stringify({version:env.VERSION,signature:env.configSignature(cfg),root:cfg.assetRoot,runId:cfg.runId,
    source:'synthetic-small-'+env.VERSION,exportName:cfg.exportName,exportFolder:cfg.exportFolder});
  if(legacy)store[pre+'_automation']={type:'Folder',properties:{automation_json:JSON.stringify({schema:'cfdepth-auto-1',identity,owner:'old',leaseUntil:0,current:null,exports:{},complete:false})}};
  function record(){const item=store[pre+'_automation_record']||store[pre+'_automation'];return item&&item.properties.automation_json?JSON.parse(item.properties.automation_json):null;}
  const data={getAsset(id){if(hooks.get)hooks.get(id);if(!store[id])throw Error('Asset not found: '+id);return clone(store[id]);},
    listAssets(root){assert.equal(root,cfg.assetRoot);return {assets:Object.keys(store).map(id=>({id}))};},
    listOperations:()=>Object.entries(tasks).map(([id,t])=>({name:id,metadata:{description:t.description,state:t.state==='COMPLETED'?'SUCCEEDED':t.state}})),
    createAsset(v,id,force,p){if(v.type==='Folder'&&p)throw Error('Extraneous field(s) present: [properties].');assert.equal(v.type,'ImageCollection');assert.equal(p,undefined);assert.equal(force,false);assert.ok(!store[id]);store[id]={type:'ImageCollection',properties:{}};},
    setAssetProperties(id,p){assert.equal(id,pre+'_automation_record');writes.push(id);store[id].properties=clone(p);if(hooks.save)hooks.save(record());},
    newTaskId:()=>['task-'+(++sequence)],
    startProcessing(id,p,cb){assert.ok(!tasks[id.replace(/-/g,'')]);tasks[id.replace(/-/g,'')]={state:'READY',description:p.description,params:p};cb({taskId:id.replace(/-/g,'')});},
    getTaskStatus(id,cb){const t=tasks[id];assert.ok(t);if(t.state==='READY'){
      if(hooks.task)hooks.task(t);
      if(t.state==='READY'){
        if(t.params.assetId){const job=t.params.element.job;add(job.stage==='prepare'?0:job.step,job.stage==='iterate'?'state':job.stage==='prepare'?'prepared':'components');
          assert.equal(t.params.assetId,pre+(job.stage==='iterate'?'_state_'+String(job.step).padStart(5,'0'):job.stage==='prepare'?'_prepared':'_components'));
          if(hooks.materialized)hooks.materialized(t.params.assetId);
        }
        t.state='COMPLETED';
      }
    }cb([t]);},
    deleteAsset(id){if(hooks.beforeDelete)hooks.beforeDelete(id);
      assert.ok(store[id]);deleted.push(id);delete store[id];if(hooks.afterDelete)hooks.afterDelete(id);}};
  function image(id){return {id,getInfo:()=>clone(store[id]),select(){return this;},rename(){return this;},addBands(){return this;}};}
  env.ee={data,Image:image,Geometry:x=>x,Dictionary:x=>({getInfo:()=>x}),Filter:{eq:()=>({})}};
  env.createGEESolver=()=>({grouped(im){const step=store[im.id].properties.step||0;return {
    map(){return this;},aggregate_histogram:()=>step>=stopAt?({'3':2,'0':2}):({'1':2,'0':2}),size:()=>4,filter:()=>({size:()=>0})};}});
  env.print=(...args)=>logs.push(args.map(String).join(' '));
  env.ui={util:{setTimeout(fn,ms){const id=++timerId;queue.push({fn,ms,id});return id;},clearTimeout(id){const i=queue.findIndex(q=>q.id===id);if(i>=0)queue.splice(i,1);}}};
  env.Date={now:()=>clock};
  env.runGEE=(config,input,runtime)=>{
    const job={stage:config.stage,step:config.step};builds.push(job);
    if(config.stage==='final'){
      // 接口对照：这里只生成确定性结果载荷，不把模拟导出当作水深算法验证。
      for(const suffix of ['','_WSE_Gradient'])runtime.emit('drive',{image:{result:suffix?'gradient':'depth',state:config.step},
        crs:grid.crs,crsTransform:grid.transform,region:{},maxPixels:1e13,folder:cfg.exportFolder,
        fileNamePrefix:cfg.exportName+suffix,formatOptions:{noData:-9999}});
    }else{
      if(config.stage==='iterate')assert.ok(store[pre+(config.step===1?'_prepared':'_state_'+String(config.step-1).padStart(5,'0'))]);
      const suffix=config.stage==='components'?'_components':config.stage==='prepare'?'_prepared':'_state_'+String(config.step).padStart(5,'0');
      runtime.emit('asset',{image:{job},assetId:pre+suffix,description:'stage',crs:grid.crs,crsTransform:grid.transform,region:{},maxPixels:1e13,pyramidingPolicy:{'.default':'sample'}});
    }
  };
  return {env,cfg,grid,pre,store,tasks,queue,logs,deleted,writes,builds,hooks,record,add,
    start(flag=cleanup){controller=env.runAutoGEE(cfg,null,{mode:'auto',pollSeconds:30,resume:true,cleanupPreviousStates:flag});return controller;},
    one(){const q=queue.shift();assert.ok(q,'Missing queued callback: '+logs.join('\n'));clock+=q.ms;q.fn();},
    until(pred,limit=300){for(let i=0;!pred();i++){assert.ok(i<limit,logs.join('\n'));this.one();}},
    finish(){this.until(()=>controller.isStopped());return record();},
    pause(){controller.stop();queue.length=0;clock+=300001;},
    stopped:()=>controller.isStopped(),states(){return Object.keys(store).filter(id=>id.startsWith(pre+'_state_'));}};
}
let n=0;function test(name,fn){fn();n++;console.log('PASS '+name);}
test('new run retains exactly latest state and exports without changing protected inputs',()=>{
  const h=simulation({saved:-1});h.start();const r=h.finish();assert.equal(r.complete,true,h.logs.join("\n"));
  assert.equal(r.latestState.step,4);assert.equal(r.cleanup.doneThrough,3);assert.deepEqual(h.states(),[h.pre+'_state_00004']);
  assert.equal(Object.keys(h.store).length,4);assert.equal(h.deleted.length,3);
  assert.deepEqual(h.builds.map(j=>j.stage+':'+j.step),['components:1','prepare:1','iterate:1','iterate:2','iterate:3','iterate:4','final:4']);
  assert.equal(Object.values(h.tasks).filter(t=>!t.params.assetId).length,2);
});
test('all-state and rolling retention execute identical stages and final export parameters',()=>{
  const a=simulation({cleanup:false}),b=simulation({cleanup:true});const originals=clone(b.store);
  a.start();a.finish();b.start();b.finish();assert.deepEqual(a.builds,b.builds);
  assert.equal(a.states().length,4);assert.equal(b.states().length,1);
  for(const suffix of ['_components','_prepared'])assert.deepEqual(clone(b.store[b.pre+suffix]),originals[b.pre+suffix]);
  assert.deepEqual(clone(Object.values(a.tasks).filter(t=>!t.params.assetId).map(t=>t.params)),clone(Object.values(b.tasks).filter(t=>!t.params.assetId).map(t=>t.params)));
});
test('migrates complete schema1 history and never requires the removed first state',()=>{
  const h=simulation({saved:4,legacy:true});h.start();const r=h.finish();assert.equal(r.schema,'cfdepth-auto-2');
  assert.equal(r.latestState.step,4);assert.equal(h.deleted.length,3);assert.deepEqual(h.states(),[h.pre+'_state_00004']);
  h.pause();h.start();h.finish();assert.equal(h.deleted.length,3);assert.equal(h.builds.length,1);
});
test('legacy holes cannot be legitimized by migration',()=>{
  const h=simulation({saved:3,legacy:true});delete h.store[h.pre+'_state_00002'];h.start();
  assert.equal(h.stopped(),true);assert.match(h.logs.at(-1),/gap/);assert.equal(h.record().schema,'cfdepth-auto-1');assert.equal(h.deleted.length,0);
});
test('migration does not promote an in-flight asset whose task was cancelled',()=>{
  const h=simulation({cleanup:false});h.start();h.until(()=>h.record().current&&h.record().current.stage==='iterate'&&h.record().current.attempted);
  h.pause();h.add(3);const old=h.record();old.schema='cfdepth-auto-1';delete old.latestState;delete old.cleanup;
  h.store[h.pre+'_automation_record'].properties.automation_json=JSON.stringify(old);
  Object.values(h.tasks)[0].state='CANCELLED';h.start(true);h.finish();
  assert.match(h.logs.at(-1),/CANCELLED/);assert.equal(h.record().latestState.step,2);assert.equal(h.deleted.length,0);
  assert.ok(h.store[h.pre+'_state_00002']);
});
test('failed next task keeps the previous verified checkpoint',()=>{
  const h=simulation();h.hooks.task=t=>{if(t.params.element.job.step===3){t.state='FAILED';t.error_message='quota exceeded';}};
  h.start();h.finish();assert.match(h.logs.at(-1),/quota exceeded/);assert.ok(h.store[h.pre+'_state_00002']);
  assert.deepEqual(h.deleted,[h.pre+'_state_00001']);assert.equal(h.record().latestState.step,2);
});
test('invalid new asset does not advance checkpoint or delete previous state',()=>{
  const h=simulation();h.hooks.materialized=id=>{if(id.endsWith('00003'))storeBad(h,id);};
  function storeBad(h,id){h.store[id].bands.pop();}
  h.start();h.finish();assert.match(h.logs.at(-1),/missing band/);assert.ok(h.store[h.pre+'_state_00002']);assert.equal(h.record().latestState.step,2);
});
test('checkpoint and deletion authorization must be read back before deleting',()=>{
  const h=simulation();let tamper=false;
  h.hooks.save=r=>{if(r.cleanup.pending.length&&!tamper){tamper=true;const corrupt=clone(r);corrupt.leaseUntil++;
    h.store[h.pre+'_automation_record'].properties.automation_json=JSON.stringify(corrupt);}};
  h.start();h.finish();assert.match(h.logs.at(-1),/readback mismatch/);assert.equal(h.deleted.length,0);assert.equal(h.states().length,2);
});
test('resume after committing cleanup but before deletion',()=>{
  const h=simulation({saved:3});h.start();h.until(()=>h.record().cleanup.pending.length===2);h.pause();h.start();h.finish();
  assert.deepEqual(h.states(),[h.pre+'_state_00004']);assert.equal(h.deleted.length,3);
});
test('response lost after successful deletion is reconciled without duplicate deletion',()=>{
  const h=simulation();let once=false;h.hooks.afterDelete=()=>{if(!once){once=true;throw Error('network response lost');}};
  h.start();h.finish();assert.equal(h.record().complete,true);assert.equal(new Set(h.deleted).size,h.deleted.length);
});
test('page closes after server deletion and before receipt save; restart recovers',()=>{
  const h=simulation({saved:3});h.hooks.afterDelete=()=>{throw Error('network response lost');};
  h.start();h.until(()=>h.deleted.length===1);assert.equal(h.record().cleanup.doneThrough,0);
  h.pause();delete h.hooks.afterDelete;h.start();h.finish();assert.equal(h.record().complete,true);assert.equal(h.deleted.length,3);
});
test('partial deletion permission failure pauses; latest remains reusable',()=>{
  const h=simulation({saved:4});h.hooks.beforeDelete=id=>{if(id.endsWith('00002'))throw Error('Permission denied');};
  h.start();h.finish();assert.match(h.logs.at(-1),/Permission denied/);assert.equal(h.record().cleanup.doneThrough,1);
  assert.ok(h.store[h.pre+'_state_00004']);assert.equal(h.builds.length,0);
  h.pause();delete h.hooks.beforeDelete;h.start();h.finish();assert.equal(h.record().complete,true);
});
test('persistent network deletion errors exhaust three retries without advancing',()=>{
  const h=simulation();let calls=0;h.hooks.beforeDelete=()=>{calls++;throw Error('network unavailable');};
  h.start();h.finish();assert.equal(calls,4);assert.equal(h.record().cleanup.doneThrough,0);
  assert.equal(h.deleted.length,0);assert.equal(h.builds.length,0);assert.ok(h.store[h.pre+'_state_00002']);
});
test('ambiguous not-found permission response cannot count as a deletion receipt',()=>{
  const h=simulation();h.start();h.until(()=>h.record().cleanup.pending.length===1);
  h.hooks.get=id=>{if(id.endsWith('_state_00001'))throw Error('Asset not found or caller does not have access');};
  h.finish();assert.match(h.logs.at(-1),/does not have access/);assert.equal(h.record().cleanup.doneThrough,0);
  assert.equal(h.deleted.length,0);assert.equal(h.builds.length,0);
});
test('missing latest is fatal before any cleanup or new task',()=>{
  const h=simulation();h.start();h.until(()=>h.record().cleanup.pending.length===1);delete h.store[h.pre+'_state_00002'];
  h.finish();assert.match(h.logs.at(-1),/Latest retained state is missing/);assert.equal(h.deleted.length,0);assert.equal(h.builds.length,0);
});
test('old token changes are rejected and unrelated runs are untouched',()=>{
  const h=simulation();h.store[h.pre+'_other_state_00001']={type:'Image',properties:{}};h.start();h.until(()=>h.record().cleanup.pending.length===1);
  h.store[h.pre+'_state_00001'].updateTime='changed';h.finish();assert.match(h.logs.at(-1),/Old state token changed/);
  assert.equal(h.deleted.length,0);assert.ok(h.store[h.pre+'_other_state_00001']);
});
test('latest token changes and wrong old source are rejected',()=>{
  for(const target of ['latest','source']){
    const h=simulation();h.start();h.until(()=>h.record().cleanup.pending.length===1);
    if(target==='latest')h.store[h.pre+'_state_00002'].updateTime='changed';else h.store[h.pre+'_state_00001'].properties.source_token='wrong';
    h.finish();assert.equal(h.deleted.length,0);assert.match(h.logs.at(-1),/token changed|source changed|source mismatch/);
  }
});
test('disabling cleanup keeps new states without demanding previously deleted states',()=>{
  const h=simulation();h.start();h.until(()=>h.record().cleanup.doneThrough===1);h.pause();h.start(false);h.finish();
  assert.equal(h.record().complete,true);assert.deepEqual(h.states(),[2,3,4].map(s=>h.pre+'_state_'+String(s).padStart(5,'0')));
  assert.equal(h.deleted.length,1);
});
test('committed cleanup finishes after disabling, but no new deletions are scheduled',()=>{
  const h=simulation({saved:3});h.start();h.until(()=>h.record().cleanup.pending.length===2);h.pause();h.start(false);h.finish();
  assert.deepEqual(h.states(),[h.pre+'_state_00003',h.pre+'_state_00004']);assert.equal(h.deleted.length,2);
});
test('new schema rejects unrelated paths malformed authorization and unrecorded gaps',()=>{
  const h=simulation();h.start();h.until(()=>h.record().cleanup.pending.length===1);const r=h.record();
  for(const change of [j=>j.cleanup.pending[0].path=h.pre+'_components',j=>j.cleanup.pending[0].step=2,
    j=>j.cleanup.through=99,j=>j.latestState.path='projects/other/assets/state_00002']){
    const x=clone(r);change(x);assert.throws(()=>h.env.validateRetentionRecord(x,h.cfg.assetRoot,h.cfg));
  }
  const x=clone(r);x.cleanup={through:0,doneThrough:0,pending:[]};
  assert.throws(()=>h.env.autoInventory([h.pre+'_components',h.pre+'_prepared',h.pre+'_state_00002'],h.cfg.assetRoot,h.cfg,x),/gap/);
  assert.equal(h.env.autoAssetMissing('not found or caller does not have access'),false);
  assert.equal(h.env.autoAssetMissing('permission denied'),false);assert.equal(h.env.autoAssetMissing('Asset not found'),true);
});
console.log(n+' retention tests passed; GEE asset deletion is simulated, NOT cloud-validated.');
