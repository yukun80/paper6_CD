'use strict';
const fs=require('node:fs'),vm=require('node:vm'),path=require('node:path'),assert=require('node:assert/strict');
const source=fs.readFileSync(path.join(__dirname,'../CFDepth_0919.txt'),'utf8');
const env={module:{exports:{}}};vm.runInNewContext(source,env);
const clone=x=>JSON.parse(JSON.stringify(x)),cfg={...env.CONFIG},root='projects/p/assets',identity='test-identity';
const current=root+'/'+cfg.runId+'_automation_record',old=root+'/'+cfg.runId+'_automation';
function initial(){return {schema:'cfdepth-auto-2',identity,current:null,exports:{},latestState:null,cleanup:{through:0,doneThrough:0,pending:[]},owner:'a'};}
function setup(){
  const assets={},hooks={},calls=[];
  const data={
    getAsset(p){if(hooks.read)hooks.read(p);if(!assets[p])throw Error('Asset not found');return clone(assets[p]);},
    createAsset(value,p,force,props){
      if(value.type==='Folder'&&props)throw Error('Extraneous field(s) present: [properties].');
      assert.equal(value.type,'ImageCollection');assert.equal(props,undefined);assert.equal(force,false);assert.ok(!assets[p]);
      calls.push('create');assets[p]={type:'ImageCollection',properties:{}};if(hooks.created)hooks.created();
    },
    setAssetProperties(p,props){assert.equal(p,current);assert.equal(assets[p].type,'ImageCollection');
      if(hooks.beforeWrite)hooks.beforeWrite();calls.push('write');Object.assign(assets[p].properties,clone(props));if(hooks.written)hooks.written();}
  };
  return {assets,hooks,calls,data,store:env.createAutomationStore(data,root,cfg,identity)};
}
let n=0;function test(name,fn){fn();n++;console.log('PASS '+name);}
test('reproduces rejected Folder properties; new creation separates properties and roundtrips',()=>{
  const h=setup();assert.throws(()=>h.data.createAsset({type:'Folder'},old,false,{automation_json:'{}'}),/Extraneous/);
  assert.equal(h.store.read(),null);const r=initial();h.store.write(r);
  assert.deepEqual(clone(h.store.read()),r);assert.deepEqual(h.calls,['create','write']);
  r.owner='b';h.store.write(r);assert.deepEqual(clone(h.store.read()),r);assert.equal(h.calls.filter(x=>x==='create').length,1);
});
test('legacy JSON migrates read-only and marker permits subsequent new journal progress',()=>{
  const h=setup(),r=initial();r.schema='cfdepth-auto-1';delete r.latestState;delete r.cleanup;
  h.assets[old]={type:'Folder',properties:{automation_json:JSON.stringify(r)}};const saved=clone(h.assets[old]);
  assert.deepEqual(clone(h.store.read()),r);h.store.write(initial());
  const next=initial();next.owner='next';h.store.write(next);
  assert.deepEqual(clone(h.store.read()),next);assert.deepEqual(h.assets[old],saved);
});
test('empty legacy Folder survives and is not treated as a record',()=>{
  const h=setup();h.assets[old]={type:'Folder',properties:{}};assert.equal(h.store.read(),null);
  h.store.write(initial());assert.deepEqual(h.assets[old],{type:'Folder',properties:{}});
});
test('two conflicting records stop; changed legacy after migration also stops',()=>{
  const h=setup(),r=initial();h.assets[old]={type:'Folder',properties:{automation_json:JSON.stringify(r)}};
  const other={...r,owner:'other'};
  h.assets[current]={type:'ImageCollection',properties:{automation_json:JSON.stringify(other)}};
  assert.throws(()=>h.store.read(),/records conflict/);assert.equal(h.calls.length,0);
  delete h.assets[current];h.store.write(r);h.assets[old].properties.automation_json=JSON.stringify(other);
  assert.throws(()=>h.store.write(r),/Legacy.*conflicts/);
});
test('wrong identities/types and malformed JSON stop before writes',()=>{
  for(const item of [{type:'Folder',properties:{}},{type:'ImageCollection',properties:{automation_json:'broken'}},
    {type:'ImageCollection',properties:{automation_json:JSON.stringify({...initial(),identity:'other'})}}]){
    const h=setup();h.assets[current]=item;assert.throws(()=>h.store.read(),/回读记录/);assert.equal(h.calls.length,0);
  }
});
test('creation response loss can resume from empty verified collection without overwrite',()=>{
  const h=setup();h.hooks.created=()=>{throw Error('network response lost');};
  assert.throws(()=>h.store.write(initial()),/创建记录资产.*network/);assert.equal(h.store.read(),null);
  delete h.hooks.created;h.store.write(initial());assert.deepEqual(clone(h.store.read()),initial());
  assert.equal(h.calls.filter(x=>x==='create').length,1);
});
test('write failure preserves empty asset and supplies specific error prefix',()=>{
  const h=setup();h.hooks.beforeWrite=()=>{throw Error('Permission denied');};
  assert.throws(()=>h.store.write(initial()),/写入记录.*Permission/);assert.equal(h.store.read(),null);
  delete h.hooks.beforeWrite;h.store.write(initial());assert.deepEqual(clone(h.store.read()),initial());
});
test('write response loss and read network failure retain recoverable record',()=>{
  const h=setup();h.hooks.written=()=>{throw Error('network response lost');};
  assert.throws(()=>h.store.write(initial()),/写入记录.*network/);delete h.hooks.written;
  assert.deepEqual(clone(h.store.read()),initial());
  h.hooks.read=()=>{throw Error('network unavailable');};assert.throws(()=>h.store.read(),/回读记录.*network/);
  delete h.hooks.read;h.store.write(initial());assert.equal(h.calls.filter(x=>x==='create').length,1);
});
test('altered readback cannot pass commit verification',()=>{
  const h=setup();h.hooks.written=()=>{const r=initial();r.owner='tampered';h.assets[current].properties.automation_json=JSON.stringify(r);};
  assert.throws(()=>h.store.write(initial()),/回读记录.*readback mismatch/);
});
test('read-only helper accepts REST collection enum and performs no writes',()=>{
  const h=setup();h.assets[current]={type:'IMAGE_COLLECTION',properties:{automation_json:JSON.stringify(initial())}};
  assert.deepEqual(clone(env.createAutomationStore(h.data,root,cfg).read()),initial());assert.equal(h.calls.length,0);
});
console.log(n+' journal interface tests passed; cloud metadata support still requires GEE acceptance.');
