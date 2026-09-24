import argparse
import json
from pathlib import Path
from .config import load_config
from .pipeline import check_inputs,prepare,solve,audit

def main():
    parser=argparse.ArgumentParser(description='Offline CCDepth reproducing frozen CFDepth rules on the original geographic grid')
    parser.add_argument('command',choices=['check','prepare','solve','run','audit'])
    parser.add_argument('--config');parser.add_argument('--run-dir');parser.add_argument('--resume',action='store_true')
    a=parser.parse_args()
    if a.command in ('check','prepare','run'):
        if not a.config:parser.error('--config required')
        cfg=load_config(a.config)
        if a.command=='check':result=check_inputs(cfg)
        elif a.command=='prepare':result=str(prepare(cfg))
        else:
            root=Path(cfg['runtime']['output_dir'])
            if not a.resume:prepare(cfg)
            result=solve(root,expected_config=cfg)
    else:
        if not a.run_dir:parser.error('--run-dir required')
        result=solve(a.run_dir) if a.command=='solve' else audit(a.run_dir)
    if isinstance(result,dict) and result.get('status')=='COMPLETE':
        output_dir=a.run_dir if a.command=='audit' else result['config']['runtime']['output_dir']
        print(f"COMPLETE: {result['solve']['components']} components; "
              f"solved {result['audit']['solved_fraction']:.2%}; audit passed. "
              f"8 TIFFs: {output_dir}")
    else:print(json.dumps(result,ensure_ascii=False,indent=2))
