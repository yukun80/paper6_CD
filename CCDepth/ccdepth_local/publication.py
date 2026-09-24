"""Publish new CCDepth TIFFs and audit both current and legacy product schemas."""
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import os
import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window
from .contracts import GridSpec
from .products import PRODUCTS
from .workspace import WorkState,close_maps


LEGACY_PRODUCTS={
 'CFDepth_WSE.tif':('float32',-9999),
 'CFDepth_depth_solved.tif':('float32',-9999),
 'CFDepth.tif':('float32',-9999),
 'CFDepth_depth_signed.tif':('float32',-9999),
 'CFDepth_WSE_gradient_solved.tif':('float32',-9999),
 'CFDepth_gradient_directions.tif':('uint8',255),
 'CFDepth_status.tif':('int8',-1),
 'CFDepth_QA.tif':('uint16',65535),
}
SCHEMAS={
    'CCDEPTH':(PRODUCTS,'CCDepth'),
    'CFDEPTH':(LEGACY_PRODUCTS,'CFDepth'),
}


def _schema_for_root(root):
    root=Path(root)
    matches=[(namespace,products,prefix) for namespace,(products,prefix) in SCHEMAS.items()
             if all((root/name).is_file() for name in products)]
    if len(matches)>1:raise ValueError('Ambiguous mixed CCDepth/CFDepth product families')
    return matches[0] if matches else None


def _names(prefix):
    return {
        'wse':f'{prefix}_WSE.tif',
        'depth':f'{prefix}_depth_solved.tif',
        'legacy':f'{prefix}.tif',
        'signed':f'{prefix}_depth_signed.tif',
        'gradient':f'{prefix}_WSE_gradient_solved.tif',
        'directions':f'{prefix}_gradient_directions.tif',
        'status':f'{prefix}_status.tif',
        'qa':f'{prefix}_QA.tif',
    }


def completed_metadata(root):
    schema=_schema_for_root(root)
    if schema is None:return None
    namespace,products,_=schema
    reports=[]
    for name in products:
        with rasterio.open(Path(root)/name) as ds:
            tags=ds.tags(ns=namespace)
            if tags.get('complete')!='true':return None
            if tags.get('product')!=name:raise ValueError('Product identity mismatch')
            reports.append(json.loads(tags['run']))
    if any(r!=reports[0] for r in reports):return None
    if reports[0].get('format')!=2:raise ValueError('Unsupported TIFF completion format')
    tagged_namespace=reports[0].get('product_namespace',namespace)
    if tagged_namespace!=namespace:raise ValueError('Product namespace mismatch')
    if set(reports[0].get('pixel_hashes',{}))!=set(products):raise ValueError('Product hash manifest mismatch')
    return reports[0]


def publish(root,cfg,products,selected,summary):
    root=Path(root);dest=root/'.work/prepared';stage=root/'.work/publish';stage.mkdir(exist_ok=True)
    with WorkState(root) as state:
        g=GridSpec(**state.get('grid'))
        report=dict(format=2,product_namespace='CCDEPTH',status='COMPLETE',whole_scene=selected is None,
                    identity=state.get('identity'),config=cfg,grid=asdict(g),
                    alignment=state.get('inputs').get('metadata',{}).get('alignment'),
                    prepare=state.get('prepare'),solve=summary)
    report['baseline']=report['identity']['baseline']
    maps={k:np.load(dest/(k+'.npy'),mmap_mode='r') for k in ('lookup','labels','input_qa')}
    hashes={}
    try:
        qa_name=_names('CCDepth')['qa']
        for name,(dtype,nodata) in PRODUCTS.items():
            digest=hashlib.sha256()
            with rasterio.open(stage/name,'w',driver='GTiff',height=g.height,width=g.width,count=1,
                               dtype=dtype,crs=g.crs,transform=Affine(*g.transform),nodata=nodata,
                               tiled=True,compress='deflate',BIGTIFF='IF_SAFER',NUM_THREADS='2') as dst:
                for y in range(0,g.height,256):
                    h=min(256,g.height-y);ix=maps['lookup'][y:y+h];valid=ix>=0
                    if selected:valid&=np.isin(maps['labels'][y:y+h],list(selected))
                    a=np.array(maps['input_qa'][y:y+h],dtype=dtype) if name==qa_name else np.full((h,g.width),nodata,dtype=dtype)
                    a[valid]=products[name][ix[valid]]
                    dst.write(a,1,window=Window(0,y,g.width,h));digest.update(a.tobytes())
            hashes[name]=digest.hexdigest()
    finally:close_maps(*maps.values())
    report['pixel_hashes']=hashes
    for name in PRODUCTS:
        with rasterio.open(stage/name,'r+') as dst:
            dst.update_tags(ns='CCDEPTH',complete='true',product=name,run=json.dumps(report,allow_nan=False))
    audit(stage)
    for name in PRODUCTS:os.replace(stage/name,root/name)
    return audit(root)


def audit(root):
    root=Path(root);report=completed_metadata(root)
    if report is None:raise ValueError('Incomplete TIFF publication; resume the run')
    schema=_schema_for_root(root)
    namespace,products,prefix=schema
    if report.get('product_namespace',namespace)!=namespace:raise ValueError('Product family does not match run metadata')
    names=_names(prefix)
    g=GridSpec(**report['grid']);counts={str(s):0 for s in (0,3,4,5)};success=positive=weak_pixels=0
    digests={name:hashlib.sha256() for name in products}
    with ExitStack() as stack:
        datasets={name:stack.enter_context(rasterio.open(root/name)) for name in products}
        for name,ds in datasets.items():
            dtype,nodata=products[name]
            if ds.shape!=g.shape or ds.crs!=rasterio.crs.CRS.from_string(g.crs) or tuple(ds.transform)[:6]!=tuple(g.transform):raise ValueError('Product grid changed')
            if ds.count!=1 or ds.dtypes[0]!=dtype or ds.nodata!=nodata:raise ValueError('Product encoding changed')
        for y in range(0,g.height,256):
            win=Window(0,y,g.width,min(256,g.height-y))
            arrays={name:ds.read(1,window=win) for name,ds in datasets.items()}
            for name,a in arrays.items():digests[name].update(a.tobytes())
            s=arrays[names['status']];ok=(s==3)|(s==4);q=arrays[names['qa']]
            if not np.isin(s,[-1,0,3,4,5]).all():raise ValueError('Nonterminal state in product')
            for role in ('wse','depth','signed'):
                a=arrays[names[role]]
                if not np.array_equal(a!=-9999,ok) or not np.isfinite(a[ok]).all():raise ValueError('Solved domain mismatch')
            depth=arrays[names['depth']];signed=arrays[names['signed']];legacy=arrays[names['legacy']]
            if not np.array_equal(depth[ok],np.maximum(signed[ok],0)):raise ValueError('Depth definition mismatch')
            valid=legacy!=-9999
            if np.any(valid&~ok) or not np.array_equal(legacy[valid],signed[valid]):raise ValueError('Legacy domain mismatch')
            directions=arrays[names['directions']];gradient=arrays[names['gradient']]
            if np.any(directions[~ok]!=255) or not np.isin(directions[ok],[0,1,2,3]).all():raise ValueError('Gradient direction mismatch')
            if not np.array_equal(gradient!=-9999,ok&(directions>0)) or np.any(gradient[gradient!=-9999]<0):raise ValueError('Gradient domain mismatch')
            if np.any(((q&16)>0)!=ok) or np.any(((q&12)>0)&ok) or np.any(q[s==0]!=4) or np.any(q[s==5]!=8):raise ValueError('QA mismatch')
            weak=(q&1024)>0
            if np.any(weak&~ok):raise ValueError('Weak boundary QA outside solved domain')
            if weak.any() and report['config']['runtime'].get('coverage_mode','baseline_v1')!='weak_boundary_v1':raise ValueError('Unexpected weak boundary estimates')
            weak_pixels+=int(weak.sum())
            for status in counts:counts[status]+=int((s==int(status)).sum())
            success+=int(ok.sum());positive+=int(valid.sum())
    if {k:v.hexdigest() for k,v in digests.items()}!=report['pixel_hashes']:raise ValueError('Product checksum mismatch')
    expected=report['solve']
    if weak_pixels!=expected.get('weak_boundary_pixels',0):raise ValueError('Weak boundary accounting mismatch')
    if counts!=expected['pixels_by_status'] or sum(expected['counts'].values())!=expected['components']:raise ValueError('Component/pixel accounting mismatch')
    if report['whole_scene'] and (sum(counts.values())!=report['prepare']['support_pixels'] or expected['components']!=report['prepare']['components']):raise ValueError('Whole-scene coverage mismatch')
    return report|dict(audit=dict(passed=True,pixels_by_status=counts,solved_pixels=success,
                                 legacy_positive_pixels=positive,solved_fraction=success/max(1,sum(counts.values()))))
