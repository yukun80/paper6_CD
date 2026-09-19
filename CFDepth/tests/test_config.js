'use strict';
// 客户端配置测试无需 GEE 凭据；不模拟服务端像元运算。
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../CFDepth_0919.txt'),'utf8');
const context={module:{exports:{}}};
vm.runInNewContext(source,context);
const api=context.module.exports;
const cfg={...api.CONFIG,assetRoot:'projects/example/assets/CFDepth_runs'};
let count=0;
function test(name,fn){fn();count++;console.log('PASS '+name);}
function imported(id){return {select(){return this;},get(key){
  assert.equal(key,'system:id');return {getInfo(){return id;}};
}};}

test('native folder string and automatic mode accepted; objects rejected',()=>{
  assert.doesNotThrow(()=>api.validateConfig(cfg));
  assert.doesNotThrow(()=>api.validateConfig({...cfg,assetRoot:''}));
  const serverString={indexOf(){throw Error('Server method must not be called');}};
  for(const root of [null,undefined,42,{},imported('image-id'),serverString,[],new String('folder')]){
    assert.throws(()=>api.validateConfig({...cfg,assetRoot:root}),/原生 JavaScript 字符串.*getFloodInput/);
  }
  for(const root of ['projects/YOUR_PROJECT/assets/CFDepth_runs','projects/<项目ID>/assets/CFDepth_runs']){
    assert.throws(()=>api.validateConfig({...cfg,assetRoot:root}),/占位路径/);
  }
});
test('missing import fails before any Earth Engine call in every production stage',()=>{
  for(const stage of ['components','prepare','iterate','final']){
    for(const input of [null,undefined,'projects/example/assets/flood',{},[]]){
      assert.throws(()=>context.runGEE({...cfg,stage},input),/缺少有效的 getFloodInput/);
    }
  }
});
test('Node and fixtures never evaluate the imported image selector',()=>{
  for(const fixture of ['small','large']){
    assert.equal(context.resolveFloodInput({...cfg,fixture}),null);
    assert.doesNotThrow(()=>api.validateFloodInput({...cfg,fixture},null));
  }
});
test('one variable replacement selects an import exactly once in every stage',()=>{
  const input=imported('projects/example/assets/flood');
  input.toJSON=()=>{throw Error('Do not serialize imported images');};
  for(const variable of ['image4','image2','image3','image']){
    let calls=0;
    const loaded={module:{exports:{}},[variable]:input,ee:{Image(value){
      calls++;assert.equal(value,input);return value;
    }}};
    vm.runInNewContext(source.replace('ee.Image(image4)','ee.Image('+variable+')'),loaded);
    assert.equal(calls,0); // 加载模块不访问导入影像。
    assert.equal(JSON.stringify(loaded.module.exports.CONFIG),JSON.stringify(api.CONFIG));
    for(const stage of ['components','prepare','iterate','final']){
      const before=calls;
      assert.equal(loaded.resolveFloodInput({...cfg,stage}),input);
      assert.equal(calls,before+1);
    }
  }
});
test('undefined imports have clear errors and unrelated Earth Engine errors propagate',()=>{
  const loaded={module:{exports:{}},ee:{Image(value){return value;}}};
  vm.runInNewContext(source,loaded);
  for(const stage of ['components','prepare','iterate','final']){
    assert.throws(()=>loaded.resolveFloodInput({...cfg,stage}),/洪水导入变量未定义.*getFloodInput/);
  }
  loaded.image4=imported('flood');
  const error=new Error('GEE image construction failed');
  loaded.ee.Image=()=>{throw error;};
  assert.throws(()=>loaded.resolveFloodInput(cfg),e=>e===error);
});
test('component preparation reads the selected import',()=>{
  const input=imported('projects/example/assets/selected-image');
  const sentinel=new Error('Reached selected input before server operations');
  context.ee={Image(value){assert.equal(value,input);throw sentinel;},
    data:{getAsset(id){return {type:id.endsWith('selected-image')?'IMAGE':'FOLDER'};}}};
  context.print=()=>{};
  assert.throws(()=>context.runGEE(cfg,input),e=>e===sentinel);
  delete context.ee;
});
test('source tokens detect asset revisions and selected image changes',()=>{
  let revision='revision-1';
  const token=id=>JSON.stringify([id,revision]);
  const input=imported('flood-a');
  const initial=api.floodSourceToken(input,cfg,token);
  assert.equal(api.floodSourceToken(input,cfg,token),initial);
  revision='revision-2';
  assert.notEqual(api.floodSourceToken(input,cfg,token),initial);
  revision='revision-1';
  assert.notEqual(api.floodSourceToken(imported('flood-b'),cfg,token),initial);
});
test('computed images still require an explicit revision',()=>{
  const input=imported(null);
  const noAsset=()=>{throw Error('Computed images do not have an asset token');};
  assert.throws(()=>api.floodSourceToken(input,cfg,noAsset),/inputRevision.*getFloodInput/);
  assert.equal(api.floodSourceToken(input,{...cfg,inputRevision:'computed-v2'},noAsset),'computed-v2');
});
test('automatic parent directory supports project, nested and legacy IDs',()=>{
  for(const id of ['projects/example/assets/flood','projects/example/assets/folder/flood',
    'users/example/flood','users/example/folder/flood',
    'projects/earthengine-legacy/assets/users/example/flood']){
    const root=id.slice(0,id.lastIndexOf('/')),calls=[];
    const location=context.resolveAssetLocation({...cfg,assetRoot:''},imported(id),path=>{
      calls.push(path);return {type:path===id?'IMAGE':'FOLDER'};
    },(path,options)=>{
      assert.equal(options.pageSize,1);calls.push(path);return {assets:[]};
    });
    assert.equal(location.root,root);assert.deepEqual(calls,[id,root]);
  }
});
test('explicit folders support computed inputs and fixtures without guessing a source',()=>{
  const getAsset=()=>({type:'FOLDER'});
  assert.equal(context.resolveAssetLocation(cfg,imported(null),getAsset).root,cfg.assetRoot);
  assert.equal(context.resolveAssetLocation({...cfg,fixture:'small'},null,getAsset).root,cfg.assetRoot);
  for(const fixture of ['', 'small']){
    assert.throws(()=>context.resolveAssetLocation({...cfg,fixture,assetRoot:'',inputRevision:'rev'},
      imported(null),getAsset),/必须显式设置 CONFIG.assetRoot/);
  }
  const id='projects/other/assets/flood';
  assert.equal(context.resolveAssetLocation(cfg,imported(id),p=>({type:p===id?'IMAGE':'FOLDER'})).root,cfg.assetRoot);
});
test('invalid paths, target types and metadata errors fail closed',()=>{
  for(const path of ['  ','projects/example/assets/','projects/example/assets/../x',
    'projects/example/assets//x','random/folder','users/example','projects/example/assets']){
    assert.throws(()=>context.checkAssetPath(path,true),/GEE asset path/);
  }
  const id='projects/example/assets/flood';
  assert.throws(()=>context.resolveAssetLocation(cfg,imported(id),()=>({type:'IMAGE'})),/expected=FOLDER/);
  assert.throws(()=>context.resolveAssetLocation(cfg,imported(id),()=>({type:'FOLDER'})),/expected=IMAGE/);
  const error=new Error('permission denied');
  assert.throws(()=>context.resolveAssetLocation(cfg,imported(id),()=>{throw error;}),e=>e===error);
});
test('auto and explicit paths and signatures retain the previous stage contract',()=>{
  const root='projects/example/assets';
  const auto={...cfg,assetRoot:''},explicit={...cfg,assetRoot:root};
  const originalParams={version:api.VERSION};
  Object.keys(explicit).forEach(k=>{
    if(!['stage','step','exportName','exportFolder','assetRoot','runId','diagnosticDetails'].includes(k))originalParams[k]=explicit[k];
  });
  assert.equal(context.configSignature(auto),JSON.stringify(originalParams));
  assert.equal(context.configSignature(explicit),JSON.stringify(originalParams));
  const paths=context.stageAssetPaths(root,auto),old=context.stageAssetPaths(root,explicit);
  assert.equal(paths.components,root+'/'+cfg.runId+'_components');
  assert.equal(paths.prepared,old.prepared);
  assert.equal(paths.state(1),root+'/'+cfg.runId+'_state_00001');
  assert.equal(paths.state(200),old.state(200));
});
test('existing assets cannot be overwritten and permission/quota failures propagate',()=>{
  const path='projects/example/assets/state';
  assert.throws(()=>context.requireNewAsset(path,()=>({type:'IMAGE'})),/Refusing existing asset/);
  assert.doesNotThrow(()=>context.requireNewAsset(path,()=>{throw Error('Asset not found');}));
  for(const reason of ['permission denied','quota exceeded','network unavailable']){
    const error=new Error(reason);
    assert.throws(()=>context.requireNewAsset(path,()=>{throw error;}),e=>e===error);
  }
});
test('Code Editor and REST types give identical locations and all stage paths',()=>{
  for(const root of ['projects/example/assets','projects/example/assets/nested','users/example',
    'projects/earthengine-legacy/assets/users/example']){
    let previous;
    for(const types of [['Image','Folder'],['IMAGE','FOLDER']]){
      const id=root+'/flood',calls=[];
      const location=context.resolveAssetLocation({...cfg,assetRoot:''},imported(id),p=>{
        calls.push(['get',p]);return {type:p===id?types[0]:types[1]};
      },(p,options)=>{assert.equal(options.pageSize,1);calls.push(['list',p]);return {};});
      assert.equal(location.root,root);
      assert.deepEqual(calls,[['get',id],[/^projects\/[^/]+\/assets$/.test(root)?'list':'get',root]]);
      const paths=context.stageAssetPaths(location.root,cfg);
      const results=[paths.components,paths.prepared,paths.state(1),paths.state(20)];
      if(previous)assert.deepEqual(results,previous);
      previous=results;
    }
  }
});
test('wrong and absent asset types report actual type, expected type and path',()=>{
  for(const expected of ['IMAGE','FOLDER']){
    for(const actual of ['ImageCollection','IMAGE_COLLECTION','Table','TABLE','Unknown','',undefined,null,42]){
      assert.throws(()=>context.requireAssetType({type:actual},'asset-path',expected),e=>
        e.message.includes('path=asset-path')&&e.message.includes('actual=')&&e.message.includes('expected='+expected));
    }
    assert.throws(()=>context.requireAssetType(null,'missing-path',expected),/actual=null/);
  }
});
test('project root list supports empty roots and propagates API failures unchanged',()=>{
  const root='projects/example/assets',input=imported(root+'/flood');
  const get=p=>{assert.equal(p,root+'/flood');return {type:'Image'};};
  for(const response of [{},{assets:[]},{assets:[{type:'IMAGE'}]}]){
    assert.equal(context.resolveAssetLocation({...cfg,assetRoot:''},input,get,()=>response).root,root);
  }
  for(const response of [null,[],{assets:'invalid'}]){
    assert.throws(()=>context.resolveAssetLocation({...cfg,assetRoot:''},input,get,()=>response),/Invalid project asset listing/);
  }
  for(const reason of ['permission denied','quota exceeded','network unavailable']){
    const error=new Error(reason);
    assert.throws(()=>context.resolveAssetLocation({...cfg,assetRoot:''},input,get,()=>{throw error;}),e=>e===error);
  }
});
console.log(`${count} configuration tests passed; GEE server execution is NOT covered.`);
