"""Windowed preparation and complete-component solving on the original grid."""
from pathlib import Path
from dataclasses import asdict
import csv
import gc
import json
import os
import time
import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window
from scipy.ndimage import label,binary_erosion,find_objects
from . import BASELINE
from .config import parameters
from .contracts import GridSpec,DomainData,Prepared
from .grid import validate_alignment
from .io_rasterio import read_inputs,valid_dem
from .boundary import boundary_arrays,component_boundary
from .topology import build_topology
from .solver import solve_component
from .products import PRODUCTS,build_products
from .provenance import file_hash,object_hash,source_hash,atomic_json
from .checkpoint import save_checkpoint,load_checkpoint
from .resources import memory_info,peak_rss,Allocator

FIELDS={'dem':'float64','hard':'bool','lower':'float64','upper':'float64','mid':'float64','beta0':'float64','beta':'float64','wet_count':'int16','dry_count':'int16','diagnostic':'uint8'}

def load_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def mmap_new(path,dtype,shape):return np.lib.format.open_memmap(path,mode='w+',dtype=dtype,shape=shape)

def check_inputs(cfg):
    r=cfg['runtime'];counts=dict(observed=0,support=0,invalid_dem=0)
    with rasterio.open(r['mask']) as m,rasterio.open(r['dem']) as d:
        grid,error=validate_alignment(m,d)
        for y in range(0,m.height,512):
            win=Window(0,y,m.width,min(512,m.height-y))
            a=m.read(1,window=win);o=m.read_masks(1,window=win)>0
            if np.any(o&(a!=0)&(a!=1)):raise ValueError('Mask contains nonbinary valid values')
            z=d.read(1,window=win);v=valid_dem(z,d.read_masks(1,window=win))
            counts['observed']+=int(o.sum());counts['support']+=int((o&(a==1)&v).sum());counts['invalid_dem']+=int((~v).sum())
        metadata=dict(mask=dict(crs=str(m.crs),transform=list(m.transform)[:6],nodata=m.nodata,dtype=m.dtypes[0]),
                      dem=dict(crs=str(d.crs),transform=list(d.transform)[:6],nodata=d.nodata,dtype=d.dtypes[0]),alignment_max_pixels=error)
    if counts['support']==0:raise ValueError('Empty support')
    return dict(grid=asdict(grid),counts=counts,metadata=metadata,hashes={k:file_hash(r[k]) for k in ('mask','dem')})

def verify_run(root):
    root=Path(root);cfg=load_json(root/'resolved_config.json');identity=load_json(root/'identity.json')
    if identity['source']!=source_hash() or identity['config']!=object_hash(cfg):raise ValueError('Run source/config identity changed')
    for key in ('mask','dem'):
        if file_hash(cfg['runtime'][key])!=identity['inputs'][key]:raise ValueError('Input hash changed')
    manifest=load_json(root/'prepared/manifest.json')
    if object_hash(manifest)!=identity['prepared']:raise ValueError('Prepared manifest identity changed')
    for name,digest in manifest['files'].items():
        if file_hash(root/'prepared'/name)!=digest:raise ValueError(f'Prepared checksum mismatch: {name}')
    return cfg,identity

def prepare(cfg):
    start=time.perf_counter();r=cfg['runtime'];root=Path(r['output_dir'])
    info=check_inputs(cfg)
    if root.exists() and any(root.iterdir()):raise FileExistsError('Use a new run directory or --resume')
    root.mkdir(parents=True,exist_ok=True);dest=root/'prepared';dest.mkdir()
    atomic_json(root/'resolved_config.json',cfg);atomic_json(root/'input_manifest.json',info);atomic_json(root/'grid.json',info['grid'])
    g=GridSpec(**info['grid']);shape=g.shape
    support=mmap_new(dest/'support.npy','bool',shape)
    qa=mmap_new(dest/'input_qa.npy','uint16',shape)
    with rasterio.open(r['mask']) as m,rasterio.open(r['dem']) as d:
        for y in range(0,g.height,512):
            win=Window(0,y,g.width,min(512,g.height-y));sl=np.s_[y:y+int(win.height),:]
            a=m.read(1,window=win);o=m.read_masks(1,window=win)>0;z=d.read(1,window=win);v=valid_dem(z,d.read_masks(1,window=win))
            support[sl]=o&(a==1)&v;qa[sl]=(~o).astype('uint16')+2*(~v).astype('uint16')
    support.flush();qa.flush()
    labels=mmap_new(dest/'labels.npy','int32',shape);n=label(support,np.ones((3,3),bool),output=labels);labels.flush()
    counts=np.bincount(labels.ravel(),minlength=n+1)
    positions=np.flatnonzero(support.ravel());ids=labels.ravel()[positions]
    order=np.argsort(ids,kind='stable');positions=positions[order];del order,ids
    np.save(dest/'positions.npy',positions);offsets=np.r_[0,np.cumsum(counts[1:])];np.save(dest/'offsets.npy',offsets)
    lookup=mmap_new(dest/'lookup.npy','int32',shape);lookup[:]=-1;lookup.ravel()[positions]=np.arange(len(positions),dtype=np.int32);lookup.flush()
    boxes=find_objects(labels);table=[dict(id=i,pixels=int(counts[i]),bbox=[b[0].start,b[0].stop,b[1].start,b[1].stop]) for i,b in enumerate(boxes,1)]
    atomic_json(dest/'components.json',table)
    arrays={name:mmap_new(dest/(name+'.npy'),dtype,(len(positions),)) for name,dtype in FIELDS.items()}
    p=parameters(cfg['algorithm']);backend=None
    if r['solver_backend']!='python_reference':
        from .kernels_numba import boundary_kernels
        backend=boundary_kernels()
    index_seconds=time.perf_counter()-start
    halo=p.pairRadius+p.peerRadius+1;tile=r['tile_size'];done=0;boundary_start=time.perf_counter()
    for y in range(0,g.height,tile):
        for x in range(0,g.width,tile):
            y1=min(y+tile,g.height);x1=min(x+tile,g.width)
            if not support[y:y1,x:x1].any():continue
            yy=max(0,y-halo);xx=max(0,x-halo);ey=min(g.height,y1+halo);ex=min(g.width,x1+halo)
            win=Window(xx,yy,ex-xx,ey-yy);inp=read_inputs(r['mask'],r['dem'],window=win,hash_inputs=False)
            lab=np.asarray(labels[yy:ey,xx:ex]);sup=lab>0
            hard=binary_erosion(sup,np.ones((3,3),bool),border_value=0)
            domain=DomainData(inp.mask_valid,sup,inp.mask_valid&~inp.flood&inp.dem_valid,hard,sup&~hard,sup&~hard,lab,[])
            first,beta=boundary_arrays(inp,domain,p,backend)
            sl=np.s_[y-yy:y1-yy,x-xx:x1-xx];valid=sup[sl];ix=lookup[y:y1,x:x1][valid]
            arrays['dem'][ix]=inp.dem[sl][valid];arrays['hard'][ix]=hard[sl][valid];arrays['beta'][ix]=beta[sl][valid]
            for k,name in enumerate(('lower','upper','mid','beta0','wet_count','dry_count','diagnostic')):arrays[name][ix]=first[sl][:,:,k][valid]
            done+=len(ix)
        if y%(tile*8)==0:print(json.dumps(dict(stage='prepare',row=y,rows=g.height,prepared_pixels=done)),flush=True)
    for a in arrays.values():a.flush()
    boundary_seconds=time.perf_counter()-boundary_start
    manifest=dict(components=n,support_pixels=int(len(positions)),files={path.name:file_hash(path) for path in sorted(dest.iterdir()) if path.is_file()})
    atomic_json(dest/'manifest.json',manifest)
    identity=dict(baseline=BASELINE,source=source_hash(),config=object_hash(cfg),inputs=info['hashes'],prepared=object_hash(manifest))
    atomic_json(root/'identity.json',identity)
    atomic_json(root/'prepare_report.json',dict(seconds=time.perf_counter()-start,index_io_seconds=index_seconds,boundary_seconds=boundary_seconds,peak_rss=peak_rss(),components=n,support_pixels=int(len(positions))))
    return root

def load_component(root,component,cfg,arrays,positions,offsets,labels,budget):
    g=GridSpec(**load_json(root/'grid.json'));cid=component['id'];begin,end=offsets[cid-1:cid+1]
    pos=positions[begin:end];y0,y1,x0,x1=component['bbox']
    local_grid=GridSpec(g.crs,g.transform,y1-y0,x1-x0,y0,x0)
    allocator=Allocator(root/'scratch'/str(cid),budget)
    t=build_topology(labels[y0:y1,x0:x1]==cid,local_grid,parameters().lambdaC,allocator)
    # Sorting is stable: compact arrays and topology both use row-major order.
    z=np.array(arrays['dem'][begin:end]);hard=np.array(arrays['hard'][begin:end]);p=parameters(cfg['algorithm'])
    from .contracts import BoundaryData
    vals={name:np.array(arrays[name][begin:end]) for name in ('lower','upper','mid','beta0','beta','wet_count','dry_count','diagnostic')}
    high=vals['beta']>=p.highWeight
    eligible=bool(high.sum()>=p.minAnchors and max(np.ptp(t.rows[high]),np.ptp(t.cols[high]))>=p.minSpanPixels) if high.any() else False
    initial=np.zeros(len(z))
    if eligible:
        initial[:]=np.sum(vals['beta']*vals['mid'])/np.sum(vals['beta']);initial[hard]=np.maximum(initial[hard],z[hard]+p.minDepth)
    b=BoundaryData(**vals,eligible=eligible,initial_S=initial)
    return Prepared(t,z,hard,b,cid),local_grid,begin,end

def solve(root):
    start=time.perf_counter();root=Path(root);cfg,identity=verify_run(root);r=cfg['runtime'];p=parameters(cfg['algorithm'])
    dest=root/'prepared';table=load_json(dest/'components.json');positions=np.load(dest/'positions.npy',mmap_mode='r');offsets=np.load(dest/'offsets.npy',mmap_mode='r');labels=np.load(dest/'labels.npy',mmap_mode='r')
    arrays={name:np.load(dest/(name+'.npy'),mmap_mode='r') for name in FIELDS}
    total,available=memory_info();budget=int(min(20*2**30,.6*available) if r['memory_gib'] is None else r['memory_gib']*2**30)
    compdir=root/'component_results';compdir.mkdir(exist_ok=True);compact=root/'compact_products';compact.mkdir(exist_ok=True)
    products={}
    for name,(dtype,nodata) in PRODUCTS.items():
        path=compact/(name+'.npy')
        products[name]=np.load(path,mmap_mode='r+') if path.exists() else mmap_new(path,dtype,(len(positions),))
    selected=set(r['component_ids']) if r['component_ids'] else None
    if selected and selected-set(c['id'] for c in table):raise ValueError('Unknown component ID')
    # Small -> medium -> largest complete component; all receive terminal states.
    ordered=sorted((c for c in table if selected is None or c['id'] in selected),key=lambda c:c['pixels'])
    logpath=root/'components.jsonl';completed={}
    if logpath.exists():
        lines=logpath.read_text().splitlines();good=[]
        for i,line in enumerate(lines):
            try:record=json.loads(line);completed[record['id']]=record;good.append(line)
            except json.JSONDecodeError:
                if i!=len(lines)-1:raise ValueError('Corrupt component journal')
        logpath.write_text('\n'.join(good)+'\n' if good else '',encoding='utf-8')
    grad_backend=None
    if r['solver_backend']!='python_reference':
        from .kernels_numba import configure_threads,gradient_kernel
        configure_threads(r['threads']);grad_backend=gradient_kernel
    pending=[]
    for number,c in enumerate(ordered):
        cid=c['id']
        if cid in completed:continue
        t0=time.perf_counter();begin,end=offsets[cid-1:cid+1]
        # Support test before any dense bounding box or graph allocation.
        b=arrays['beta'][begin:end];high=b>=p.highWeight;pos=positions[begin:end]
        eligible=high.sum()>=p.minAnchors
        if eligible:
            ys=pos[high]//labels.shape[1];xs=pos[high]%labels.shape[1]
            eligible=max(np.ptp(ys),np.ptp(xs))>=p.minSpanPixels
        if not eligible:
            for name,(_,nodata) in PRODUCTS.items():products[name][begin:end]=nodata
            products['CFDepth_status.tif'][begin:end]=0;products['CFDepth_QA.tif'][begin:end]=4
            record=dict(id=cid,pixels=c['pixels'],status=0,reason='insufficient_boundary_support',sweeps=0,seconds=time.perf_counter()-t0)
        else:
            model,grid,begin,end=load_component(root,c,cfg,arrays,positions,offsets,labels,budget)
            checkpoint=root/'checkpoint'/str(cid);ckid=identity|dict(component_id=cid,backend=r['solver_backend'])
            saved=load_checkpoint(checkpoint,ckid);last=[time.monotonic()]
            def callback(payload,actions):
                now=time.monotonic()
                if now-last[0]>=r['checkpoint_seconds'] or actions['saveBase'] or actions['restoreBase'] or payload['state']['status'] not in (1,2):
                    save_checkpoint(checkpoint,payload,ckid);last[0]=now
                    print(json.dumps(dict(stage='solve',component=cid,pixels=c['pixels'],state=payload['state'],total_sweeps=payload['total_sweeps'])),flush=True)
            result=solve_component(model,p,r['solver_backend'],saved,callback)
            out=build_products(result,model,grid,p,grad_backend)
            for name,a in out.items():products[name][begin:end]=a
            record=dict(id=cid,pixels=c['pixels'],status=result.state.status,reason=result.reason,sweeps=result.total_sweeps,seconds=time.perf_counter()-t0,audit=result.audit,state=asdict(result.state))
            atomic_json(compdir/(str(cid)+'.json'),record|dict(history=result.history))
            del model,result,out
        # Flush arrays before committing the component journal record.
        pending.append(record)
        if len(pending)>=100 or number==len(ordered)-1:
            for a in products.values():a.flush()
            with logpath.open('a',encoding='utf-8') as f:
                for item in pending:f.write(json.dumps(item,allow_nan=False)+'\n')
                f.flush();os.fsync(f.fileno())
            pending=[]
        completed[cid]=record
        if number%1000==0:print(json.dumps(dict(stage='components',completed=number+1,total=len(ordered),last=record)),flush=True)
    for a in products.values():a.flush()
    rows=[completed[c['id']] for c in ordered]
    with (root/'components.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=['id','pixels','status','reason','sweeps','seconds','primary','residual','hard','mu'])
        writer.writeheader()
        for row in rows:writer.writerow({k:row.get(k,row.get('audit',{}).get(k,'')) for k in writer.fieldnames})
    atomic_json(root/'solve_report.json',dict(seconds=time.perf_counter()-start,peak_rss=peak_rss(),memory_budget=budget,components=len(rows),counts={str(s):sum(row['status']==s for row in rows) for s in (0,3,4,5)},pixels_by_status={str(s):sum(row['pixels'] for row in rows if row['status']==s) for s in (0,3,4,5)}))
    t_publish=time.perf_counter();publish(root,cfg,products,selected)
    solve_report=load_json(root/'solve_report.json');solve_report['publish_seconds']=time.perf_counter()-t_publish
    atomic_json(root/'solve_report.json',solve_report)
    return audit(root)

def publish(root,cfg,products,selected):
    g=GridSpec(**load_json(root/'grid.json'));dest=root/'prepared';lookup=np.load(dest/'lookup.npy',mmap_mode='r');labels=np.load(dest/'labels.npy',mmap_mode='r');input_qa=np.load(dest/'input_qa.npy',mmap_mode='r')
    hashes={}
    for name,(dtype,nodata) in PRODUCTS.items():
        temp=root/(name+'.tmp.tif');path=root/name
        with rasterio.open(temp,'w',driver='GTiff',height=g.height,width=g.width,count=1,dtype=dtype,crs=g.crs,transform=Affine(*g.transform),nodata=nodata,tiled=True,compress='deflate',BIGTIFF='IF_SAFER',NUM_THREADS='2') as dst:
            for y in range(0,g.height,256):
                h=min(256,g.height-y);ix=lookup[y:y+h];valid=ix>=0
                if selected:valid&=np.isin(labels[y:y+h],list(selected))
                a=np.array(input_qa[y:y+h],dtype=dtype) if name=='CFDepth_QA.tif' else np.full((h,g.width),nodata,dtype=dtype)
                a[valid]=products[name][ix[valid]]
                dst.write(a,1,window=Window(0,y,g.width,h))
        os.replace(temp,path);hashes[name]=file_hash(path)
    atomic_json(root/'products_manifest.json',hashes)

def audit(root):
    root=Path(root);cfg,identity=verify_run(root);g=GridSpec(**load_json(root/'grid.json'));hashes=load_json(root/'products_manifest.json')
    for name,digest in hashes.items():
        if file_hash(root/name)!=digest:raise ValueError('Product hash mismatch')
        with rasterio.open(root/name) as ds:
            if ds.shape!=g.shape or ds.crs!=rasterio.crs.CRS.from_string(g.crs) or tuple(ds.transform)[:6]!=tuple(g.transform):raise ValueError('Product grid changed')
    counts={s:0 for s in (0,3,4,5)};success=0;positive=0
    with rasterio.open(root/'CFDepth_status.tif') as st,rasterio.open(root/'CFDepth_WSE.tif') as wse,rasterio.open(root/'CFDepth_depth_solved.tif') as dep,rasterio.open(root/'CFDepth.tif') as legacy,rasterio.open(root/'CFDepth_QA.tif') as qa:
        for y in range(0,g.height,512):
            win=Window(0,y,g.width,min(512,g.height-y));s=st.read(1,window=win);ok=(s==3)|(s==4)
            if not np.isin(s,[-1,0,3,4,5]).all():raise ValueError('Nonterminal state in product')
            z=wse.read(1,window=win);d=dep.read(1,window=win);l=legacy.read(1,window=win);q=qa.read(1,window=win)
            if not np.array_equal(z!=-9999,ok) or not np.array_equal(d!=-9999,ok):raise ValueError('Solved domain mismatch')
            if np.any(d[ok]<0) or not np.isfinite(z[ok]).all() or np.any((l!=-9999)&~ok):raise ValueError('Invalid solved values')
            if np.any(((q&16)>0)!=ok) or np.any(((q&12)>0)&ok):raise ValueError('QA mismatch')
            for state in counts:counts[state]+=int((s==state).sum())
            success+=int(ok.sum());positive+=int((l!=-9999).sum())
    expected=load_json(root/'solve_report.json')
    if counts!={int(k):v for k,v in expected['pixels_by_status'].items()}:raise ValueError('Component/pixel accounting mismatch')
    if cfg['runtime']['component_ids'] is None:
        if sum(counts.values())!=load_json(root/'input_manifest.json')['counts']['support']:raise ValueError('Whole-scene support coverage mismatch')
        if expected['components']!=load_json(root/'prepared/manifest.json')['components']:raise ValueError('Missing components')
    result=dict(status='COMPLETE',whole_scene=cfg['runtime']['component_ids'] is None,baseline=BASELINE,identity=identity,grid=asdict(g),product_hashes=hashes,prepare=load_json(root/'prepare_report.json'),solve=expected,
                audit=dict(passed=True,pixels_by_status=counts,solved_pixels=success,legacy_positive_pixels=positive,solved_fraction=success/max(1,sum(counts.values()))))
    atomic_json(root/'run.json',result)
    return result
