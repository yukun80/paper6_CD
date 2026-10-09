"""整景统计测试：共同掩膜、网格、错误输入和写出保护。"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin

import evaluate_scene_comparison as ev


class SceneTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def raster(self, name, values, nodata, **kwargs):
        path = self.root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        values = np.array(values, dtype='uint8')
        profile = dict(driver='GTiff', height=values.shape[0], width=values.shape[1],
                       count=1, dtype='uint8', nodata=nodata,
                       crs='EPSG:32649', transform=from_origin(100, 200, 10, 10))
        profile.update(kwargs)
        with rasterio.open(path, 'w', **profile) as ds:
            ds.write(values, 1)
        return path

    def test_confusion_common_mask_and_block_sizes(self):
        gt = self.raster('gt.tif', [[1, 0, 0, 1, 3, 1, 0]], 3)
        p = self.raster('p.tif', [[1, 0, 1, 0, 1, 1, 1]], 255)
        q = self.raster('q.tif', [[1, 0, 1, 0, 0, 255, 0]], 255)
        with rasterio.open(q, 'r+') as ds:
            ds.write_mask(np.array([[255,255,255,255,255,255,0]], dtype='uint8'))
        preds = [('P','p',p), ('Q','q',q)]
        rows = ev.evaluate_region('x',gt,preds,1)
        self.assertEqual(rows, ev.evaluate_region('x',gt,preds,512))
        for r in rows:
            self.assertEqual([r[k] for k in ['TP','TN','FP','FN']], [1,1,1,1])
            self.assertEqual(r['valid_pixels'],4)
            self.assertEqual(r['excluded_pixels'],3)
            self.assertAlmostEqual(r['IoU'],1/3)
            for k in ['Precision','Recall','F1','OA']:
                self.assertEqual(r[k],.5)
        ev.write_tsv(self.root/'out.tsv',rows)
        self.assertEqual(len((self.root/'out.tsv').read_text().splitlines()),3)

    def test_zero_denominators(self):
        m = ev.metrics(0,10,0,0)
        self.assertEqual(m['OA'],1)
        for key in ['IoU','Precision','Recall','F1']:
            self.assertTrue(np.isnan(m[key]))
        self.assertEqual(ev.metrics(0,10,0,1)['Recall'],0)

    def test_invalid_inputs(self):
        gt = self.raster('gt.tif',[[0,1]],3)
        for name,values,opts in [
            ('unknown',[[0,2]],{}), ('shape',[[0]],{}),
            ('crs',[[0,1]],{'crs':'EPSG:4326'}),
            ('grid',[[0,1]],{'transform':from_origin(101,200,10,10)}),
            ('empty',[[255,255]],{})]:
            p = self.raster(name+'.tif',values,255,**opts)
            with self.subTest(name=name), self.assertRaises(ValueError):
                ev.evaluate_region('x',gt,[('P','p',p)])
        with self.assertRaises(FileNotFoundError):
            ev.evaluate_region('x',gt,[('P','p',self.root/'missing.tif')])
        gt_bad = self.raster('gt_bad.tif',[[0,2]],3)
        p = self.raster('good.tif',[[0,1]],255)
        with self.assertRaises(ValueError):
            ev.evaluate_region('x',gt_bad,[('P','p',p)])
        p = self.raster('tiny_shift.tif',[[0,1]],255,
                        transform=from_origin(100+1e-7,200,10,10))
        self.assertEqual(ev.evaluate_region('x',gt,[('P','p',p)])[0]['OA'],1)

    def test_main_preflight_protects_existing_outputs(self):
        gt = self.raster('gt.tif',[[0,1]],3)
        self.raster('pred/p/one/mosaic/change_binary.tif',[[0,1]],255)
        target = self.root/'out/scene_metrics_one.tsv'
        target.parent.mkdir(); target.write_text('old')
        with patch.multiple(ev, REGIONS={'one':str(gt),'two':str(gt)},
                            MODELS=(('P','p'),), OPENCD_ROOT=str(self.root/'pred')):
            args = ['--output-dir',str(target.parent)]
            self.assertEqual(ev.main(args),1)
            self.assertEqual(target.read_text(),'old')
            self.assertEqual(ev.main(args+['--regions','one','--dry-run']),0)
            self.assertEqual(target.read_text(),'old')
            self.assertEqual(ev.main(args+['--regions','one']),0)
            self.assertIn('IoU',target.read_text())


if __name__ == '__main__':
    unittest.main()
