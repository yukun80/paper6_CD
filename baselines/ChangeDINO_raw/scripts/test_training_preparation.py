"""统计尺度、目录保护、种子和 checkpoint 元数据回归。"""
import ast
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import prepare_training as p


class PreparationTests(unittest.TestCase):
    def test_statistics_and_pairing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ('train','val'):
                for part, value in [('A',10),('B',30),('label',255)]:
                    path = root/split/part; path.mkdir(parents=True)
                    shape = (256,256) if part=='label' else (256,256,3)
                    Image.fromarray(np.full(shape, value if split=='train' else 255, np.uint8)).save(path/'a.png')
            snapshot, stats = p.inspect_data(root)
            np.testing.assert_allclose(stats['recommended_config_fields']['mean'], [20/255]*3)
            np.testing.assert_allclose(stats['recommended_config_fields']['std'], [10/255]*3)
            self.assertEqual(stats['num_images'],2)
            self.assertEqual(stats['pixel_count'],2*256*256)
            self.assertFalse((root/'channel_stats_s1gfloods_train.json').exists())
            Image.fromarray(np.ones((256,256),np.uint8)).save(root/'val/label/a.png')
            with self.assertRaisesRegex(ValueError,'0/255'): p.inspect_data(root)
            (root/'train/B/a.png').unlink()
            with self.assertRaisesRegex(ValueError,'mismatched'): p.members(root)

    def test_run_protection_and_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=p.parse_args(['full-train','--checkpoint-root',tmp,'--run-name','chosen'])
            self.assertEqual((args.batch_size,args.num_epochs,args.lr,args.seed),(12,100,1e-4,42))
            path=p.run_path(args); path.mkdir()
            with self.assertRaises(FileExistsError): p.run_path(args)
            args.run_name=None
            self.assertNotEqual(p.run_path(args),path)
            args.mode='smoke-train'
            self.assertTrue(p.run_path(args).name.startswith('smoke-'))

    def test_seed_precedes_initialization(self):
        tree=ast.parse((p.PROJECT/'trainval.py').read_text())
        main=tree.body[-1]
        calls=[node for node in ast.walk(main) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)]
        seed=next(c for c in calls if c.func.id=='setup_seed')
        model=next(c for c in calls if c.func.id=='Trainval')
        self.assertLess(seed.lineno,model.lineno)
        self.assertEqual(ast.unparse(seed.keywords[0].value),'opt.seed')

    def test_metadata(self):
        sys.path.insert(0,str(p.PROJECT))
        from model.create_ChangeDINO import Model
        opt=SimpleNamespace(mean=[.5]*3,std=[.2]*3,stats_file='stats.json',dataset='S1GFloods',
                            dataroot='datasets',seed=42,backbone='mobilenetv2',fpn_channels=128,
                            deform_groups=4,gamma_mode='SE',beta_mode='contextgatedconv',n_layers=[1]*4,
                            dino_arch='dinov3_vits16',dino_weight='dino.pth',extract_ids=[2,5,8,11])
        meta=Model._build_checkpoint_meta(SimpleNamespace(opt=opt))
        self.assertEqual(meta['normalization']['mean'],opt.mean)
        self.assertEqual(meta['seed'],42)
        self.assertEqual(meta['dataset']['name'],'S1GFloods')
        self.assertEqual(meta['model_config']['dino_arch'],'dinov3_vits16')


if __name__=='__main__': unittest.main()
