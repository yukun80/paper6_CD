"""新批次按汇总主指标选择；旧批次保留历史回退行为。"""
import csv
from pathlib import Path
import sys


def select_checkpoint(work_dir: str | Path) -> Path:
    work = Path(work_dir).resolve()
    summary = work.parent / 'summary.tsv'
    if summary.is_file():
        with summary.open() as stream:
            rows = list(csv.DictReader(stream, delimiter='\t'))
        if rows and 'primary_metric' in rows[0]:
            matches = [r for r in rows if r['model_tag'] == work.name]
            if len(matches) != 1 or matches[0]['status'] != 'success':
                raise ValueError(f'No successful primary checkpoint record: {work}')
            row = matches[0]
            path = Path(row['best_ckpt'])
            if (not path.is_file() or path.resolve().parent != work or
                    not path.name.startswith(f"best_{row['primary_metric']}_")):
                raise ValueError(f'Invalid primary checkpoint: {path}')
            return path
    if list(work.glob('best_FloodIoU*.pth')):
        raise ValueError(f'FloodIoU batch requires an unambiguous summary: {work}')
    candidates = sorted(work.glob('best_*.pth'))
    if candidates:
        return candidates[0]
    if (work / 'latest.pth').is_file():
        return work / 'latest.pth'
    raise ValueError(f'No checkpoint found in {work}')


if __name__ == '__main__':
    try:
        print(select_checkpoint(sys.argv[1]))
    except (ValueError, OSError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
