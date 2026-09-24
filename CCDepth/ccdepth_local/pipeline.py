"""Windowed preparation and complete-component solving on the original grid."""
from pathlib import Path
from dataclasses import asdict
import time
import uuid
import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import label,binary_erosion,find_objects
from . import BASELINE
from .config import parameters
from .contracts import GridSpec,DomainData,Prepared
from .grid import validate_alignment,alignment_report
from .io_rasterio import read_inputs,valid_dem
from .boundary import boundary_arrays
from .topology import build_topology
from .solver import solve_component
from .products import PRODUCTS,build_products
from .provenance import file_hash,object_hash,source_hash
from .checkpoint import save_checkpoint,load_checkpoint
from .resources import memory_info,peak_rss,Allocator
from .workspace import WorkState,close_maps,remove_work_path
from .publication import publish,audit,completed_metadata
from .coverage import eligibility,WEAK_BOUNDARY_QA

FIELDS={'dem':'float64','hard':'bool','lower':'float64','upper':'float64','mid':'float64','beta':'float64'}

def read_state(root,key):
    with WorkState(root) as state:return state.get(key)

def mmap_new(path,dtype,shape):return np.lib.format.open_memmap(path,mode='w+',dtype=dtype,shape=shape)

def check_inputs(cfg):
    r=cfg['runtime'];counts=dict(observed=0,support=0,invalid_dem=0)
    with rasterio.open(r['mask']) as m,rasterio.open(r['dem']) as d:
        alignment=alignment_report(m,d,r.get('alignment_tolerance_pixels',1e-5),r.get('alignment_mode','strict_v1'))
        grid=alignment['grid'];error=alignment['max_pixels']
        alignment=dict(alignment);alignment['grid']=asdict(alignment['grid'])
        for y in range(0,m.height,512):
            win=Window(0,y,m.width,min(512,m.height-y))
            a=m.read(1,window=win);o=m.read_masks(1,window=win)>0
            if np.any(o&(a!=0)&(a!=1)):raise ValueError('Mask contains nonbinary valid values')
            z=d.read(1,window=win);v=valid_dem(z,d.read_masks(1,window=win))
            counts['observed']+=int(o.sum());counts['support']+=int((o&(a==1)&v).sum());counts['invalid_dem']+=int((~v).sum())
        metadata=dict(mask=dict(crs=str(m.crs),transform=list(m.transform)[:6],nodata=m.nodata,dtype=m.dtypes[0]),
                      dem=dict(crs=str(d.crs),transform=list(d.transform)[:6],nodata=d.nodata,dtype=d.dtypes[0]),
                      alignment=alignment,alignment_max_pixels=error)
    if counts['support']==0:raise ValueError('Empty support')
    return dict(grid=asdict(grid),counts=counts,metadata=metadata,hashes={k:file_hash(r[k]) for k in ('mask','dem')})

def verify_run(root):
    root=Path(root)
    with WorkState(root) as state:
        cfg=state.get('config');identity=state.get('identity');manifest=state.get('prepared')
        if object_hash(state.get('grid'))!=identity['grid'] or object_hash(state.get('table'))!=identity['table']:raise ValueError('Prepared metadata identity changed')
    if identity['source']!=source_hash() or identity['config']!=object_hash(cfg):raise ValueError('Run source/config identity changed')
    for key in ('mask','dem'):
        if file_hash(cfg['runtime'][key])!=identity['inputs'][key]:raise ValueError('Input hash changed')
    if object_hash(manifest)!=identity['prepared']:raise ValueError('Prepared manifest identity changed')
    for name,digest in manifest['files'].items():
        if file_hash(root/'.work/prepared'/name)!=digest:raise ValueError(f'Prepared checksum mismatch: {name}')
    return cfg,identity

def prepare(cfg):
    start=time.perf_counter();r=cfg['runtime'];root=Path(r['output_dir']).resolve()
    info=check_inputs(cfg)
    if root.exists() and any(root.iterdir()):raise FileExistsError('Use a new run directory or --resume')
    root.mkdir(parents=True,exist_ok=True);dest=root/'.work/prepared';dest.mkdir(parents=True)
    with WorkState(root,create=True) as state:
        state.put('config',cfg);state.put('inputs',info);state.put('grid',info['grid'])
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
    with WorkState(root) as state:state.put('table',table)
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
            win=Window(xx,yy,ex-xx,ey-yy);inp=read_inputs(r['mask'],r['dem'],config=cfg,window=win,hash_inputs=False)
            lab=np.asarray(labels[yy:ey,xx:ex]);sup=lab>0
            hard=binary_erosion(sup,np.ones((3,3),bool),border_value=0)
            domain=DomainData(inp.mask_valid,sup,inp.mask_valid&~inp.flood&inp.dem_valid,hard,sup&~hard,sup&~hard,lab,[])
            first,beta=boundary_arrays(inp,domain,p,backend)
            sl=np.s_[y-yy:y1-yy,x-xx:x1-xx];valid=sup[sl];ix=lookup[y:y1,x:x1][valid]
            arrays['dem'][ix]=inp.dem[sl][valid];arrays['hard'][ix]=hard[sl][valid];arrays['beta'][ix]=beta[sl][valid]
            for k,name in enumerate(('lower','upper','mid')):arrays[name][ix]=first[sl][:,:,k][valid]
            done+=len(ix)
        if y%(tile*8)==0:print(f'Prepare: row {y}/{g.height}, {done} supported pixels',flush=True)
    for a in arrays.values():a.flush()
    boundary_seconds=time.perf_counter()-boundary_start
    manifest=dict(components=n,support_pixels=int(len(positions)),files={path.name:file_hash(path) for path in sorted(dest.iterdir()) if path.is_file()})
    identity=dict(format=2,run_id=uuid.uuid4().hex,baseline=BASELINE,source=source_hash(),config=object_hash(cfg),inputs=info['hashes'],prepared=object_hash(manifest),table=object_hash(table),grid=object_hash(info['grid']))
    with WorkState(root) as state:
        state.put('prepared',manifest);state.put('identity',identity)
        state.put('prepare',dict(seconds=time.perf_counter()-start,index_io_seconds=index_seconds,boundary_seconds=boundary_seconds,peak_rss=peak_rss(),components=n,support_pixels=int(len(positions))))
    close_maps(support,qa,labels,lookup,*arrays.values())
    return root

def load_component(root,component,cfg,arrays,positions,offsets,labels,budget):
    g=GridSpec(**read_state(root,'grid'));cid=component['id'];begin,end=offsets[cid-1:cid+1]
    y0,y1,x0,x1=component['bbox']
    local_grid=GridSpec(g.crs,g.transform,y1-y0,x1-x0,y0,x0)
    allocator=Allocator(root/'.work/scratch'/str(cid),budget)
    t=build_topology(labels[y0:y1,x0:x1]==cid,local_grid,parameters().lambdaC,allocator)
    # Sorting is stable: compact arrays and topology both use row-major order.
    z=np.array(arrays['dem'][begin:end]);hard=np.array(arrays['hard'][begin:end]);p=parameters(cfg['algorithm'])
    from .contracts import BoundaryData
    vals={name:np.array(arrays[name][begin:end]) for name in ('lower','upper','mid','beta')}
    eligible,_=eligibility(vals['beta'],t.rows,t.cols,p,cfg['runtime'].get('coverage_mode','baseline_v1'))
    initial=np.zeros(len(z))
    if eligible:
        initial[:]=np.sum(vals['beta']*vals['mid'])/np.sum(vals['beta']);initial[hard]=np.maximum(initial[hard],z[hard]+p.minDepth)
    b=BoundaryData(**vals,beta0=None,wet_count=None,dry_count=None,diagnostic=None,eligible=eligible,initial_S=initial)
    return Prepared(t,z,hard,b,cid),local_grid,begin,end

def product_digest(products,begin,end):
    import hashlib
    digest=hashlib.sha256()
    for name,a in products.items():digest.update(a[begin:end].tobytes())
    return digest.hexdigest()


def cleanup(root):
    try:remove_work_path(root,root/'.work')
    except OSError as exc:raise RuntimeError('TIFFs are complete, but .work cleanup failed; use --resume to retry cleanup') from exc


def solve(root,expected_config=None):
    root=Path(root).resolve();complete=completed_metadata(root)
    if complete is not None:
        cfg=complete['config'];identity=complete['identity']
        if expected_config is not None and expected_config!=cfg:raise ValueError('Resume config mismatch')
        if identity['source']!=source_hash() or identity['config']!=object_hash(cfg):raise ValueError('Run source/config identity changed')
        for key in ('mask','dem'):
            if file_hash(cfg['runtime'][key])!=identity['inputs'][key]:raise ValueError('Input hash changed')
        result=audit(root);cleanup(root);return result
    start=time.perf_counter();cfg,identity=verify_run(root)
    if expected_config is not None and expected_config!=cfg:raise ValueError('Resume config mismatch')
    r=cfg['runtime'];p=parameters(cfg['algorithm']);dest=root/'.work/prepared'
    table=read_state(root,'table')
    if object_hash(table)!=identity['table']:raise ValueError('Component table identity changed')
    opened=[];products={}
    def load(path,mode='r'):
        a=np.load(path,mmap_mode=mode);opened.append(a);return a
    try:
        positions=load(dest/'positions.npy');offsets=load(dest/'offsets.npy');labels=load(dest/'labels.npy')
        arrays={name:load(dest/(name+'.npy')) for name in FIELDS}
        _,available=memory_info();budget=int(min(20*2**30,.6*available) if r['memory_gib'] is None else r['memory_gib']*2**30)
        compact=root/'.work/products';compact.mkdir(exist_ok=True)
        with WorkState(root) as state:
            completed=state.completed()
            for name,(dtype,nodata) in PRODUCTS.items():
                path=compact/(name+'.npy')
                if completed and not path.exists():raise ValueError('Missing committed product data')
                if path.exists():a=load(path,'r+')
                else:
                    a=mmap_new(path,dtype,(len(positions),));a[:]=nodata;opened.append(a)
                if a.dtype!=np.dtype(dtype) or a.shape!=(len(positions),):raise ValueError('Invalid compact product')
                products[name]=a
            selected=set(r['component_ids']) if r['component_ids'] else None
            if selected and selected-set(c['id'] for c in table):raise ValueError('Unknown component ID')
            ordered=sorted((c for c in table if selected is None or c['id'] in selected),key=lambda c:c['pixels'])
            byid={c['id']:c for c in ordered}
            for cid,record in completed.items():
                if cid not in byid or record['status'] not in (0,3,4,5) or record['pixels']!=byid[cid]['pixels']:raise ValueError('Invalid committed component')
                if record['status'] in (3,4) and not record.get('audit',{}).get('passed'):raise ValueError('Committed component audit rejected')
                begin,end=offsets[cid-1:cid+1]
                if record['digest']!=product_digest(products,begin,end):raise ValueError('Committed product checksum mismatch')
                remove_work_path(root,root/'.work/checkpoint'/str(cid))
            grad_backend=None
            if r['solver_backend']!='python_reference':
                from .kernels_numba import configure_threads,gradient_kernel
                configure_threads(r['threads']);grad_backend=gradient_kernel
            pending=[]
            def commit():
                if not pending:return
                for a in products.values():a.flush()
                state.commit_components(pending)
                for row in pending:remove_work_path(root,root/'.work/checkpoint'/str(row['id']))
                pending.clear()
            last_progress=time.monotonic()
            for number,c in enumerate(ordered):
                cid=c['id']
                if cid in completed:continue
                t0=time.perf_counter();begin,end=offsets[cid-1:cid+1]
                b=arrays['beta'][begin:end];pos=positions[begin:end]
                eligible,weak=eligibility(b,pos//labels.shape[1],pos%labels.shape[1],p,r.get('coverage_mode','baseline_v1'))
                saved_active=False
                if not eligible:
                    for name,(_,nodata) in PRODUCTS.items():products[name][begin:end]=nodata
                    products['CCDepth_status.tif'][begin:end]=0;products['CCDepth_QA.tif'][begin:end]=4
                    record=dict(id=cid,pixels=c['pixels'],status=0,reason='insufficient_boundary_support',sweeps=0,seconds=time.perf_counter()-t0)
                else:
                    model=None
                    try:
                        model,grid,begin,end=load_component(root,c,cfg,arrays,positions,offsets,labels,budget)
                        checkpoint=root/'.work/checkpoint'/str(cid);ckid=identity|dict(component_id=cid,backend=r['solver_backend'])
                        saved=load_checkpoint(checkpoint,ckid);last=time.monotonic();saved_active=saved is not None
                        def callback(payload,actions):
                            nonlocal last,saved_active,last_progress
                            now=time.monotonic()
                            if now-last_progress>=30:
                                print(f'Solve: component {cid}, {payload["total_sweeps"]} sweeps, state {payload["state"]["status"]}',flush=True)
                                last_progress=now
                            if payload['state']['status'] not in (1,2):return
                            if now-last>=r['checkpoint_seconds'] or (saved_active and (actions['saveBase'] or actions['restoreBase'])):
                                save_checkpoint(checkpoint,payload,ckid);last=now;saved_active=True
                        result=solve_component(model,p,r['solver_backend'],saved,callback)
                        out=build_products(result,model,grid,p,grad_backend)
                        if weak and result.audit['passed']:out['CCDepth_QA.tif']|=WEAK_BOUNDARY_QA
                        for name,a in out.items():products[name][begin:end]=a
                        record=dict(id=cid,pixels=c['pixels'],status=result.state.status,reason=result.reason,sweeps=result.total_sweeps,seconds=time.perf_counter()-t0,audit=result.audit)
                        record['weak_boundary']=bool(weak and result.audit['passed'])
                        del result,out,saved
                    finally:
                        if model is not None:close_maps(model.topology.neighbors)
                        model=None
                        remove_work_path(root,root/'.work/scratch'/str(cid))
                record['digest']=product_digest(products,begin,end);pending.append(record);completed[cid]=record
                if len(pending)>=100 or saved_active:commit()
                if time.monotonic()-last_progress>=30 or number==len(ordered)-1:
                    print(f'Solve: {number+1}/{len(ordered)} components',flush=True);last_progress=time.monotonic()
            commit()
            rows=[completed[c['id']] for c in ordered]
            from collections import Counter
            summary=dict(seconds=time.perf_counter()-start,peak_rss=peak_rss(),memory_budget=budget,components=len(rows),
                         weak_boundary_pixels=sum(row['pixels'] for row in rows if row.get('weak_boundary')),
                         reasons=dict(Counter(row['reason'] for row in rows)),
                         counts={str(s):sum(row['status']==s for row in rows) for s in (0,3,4,5)},
                         pixels_by_status={str(s):sum(row['pixels'] for row in rows if row['status']==s) for s in (0,3,4,5)})
        result=publish(root,cfg,products,selected,summary)
    finally:close_maps(*opened)
    cleanup(root)
    return result
