"""Run a historical analysis/acquisition script against an explicit working copy.

The historical __file__-relative project root is redirected without changing
the archived source. This is not a promise that every exploratory script is
portable: read its inputs, hardware and CLI arguments first.
"""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',type=Path,required=True)
    ap.add_argument('script')
    ap.add_argument('script_args',nargs=argparse.REMAINDER)
    a=ap.parse_args();project=a.project_root.resolve()
    snapshot=ROOT/'data/derived/project_tree'
    if project==snapshot or snapshot in project.parents:
        ap.error('Use a separate working copy; the archived data snapshot is read-only by convention.')
    source=ROOT/'code/provenance/scripts'/Path(a.script).name
    if source.suffix!='.py' or not source.exists():ap.error('Unknown archived Python script.')
    project.mkdir(parents=True,exist_ok=True);(project/'scripts').mkdir(exist_ok=True)
    # Several archived scripts copy their own provenance: stage all first-party
    # sources, never model weights or credentials.
    import shutil
    for p in source.parent.glob('*.py'):shutil.copy2(p,project/'scripts'/p.name)
    sys.path.insert(0,str(project/'scripts'))
    sys.argv=[str(project/'scripts'/source.name)]+a.script_args
    s=source.read_text(encoding='utf-8-sig')
    exec(compile(s,str(project/'scripts'/source.name),'exec'),{'__name__':'__main__','__file__':str(project/'scripts'/source.name)})
if __name__=='__main__':main()
