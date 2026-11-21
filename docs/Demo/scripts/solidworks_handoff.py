"""Use the plugin's documented standalone MCP workflow after stale add-in pipe.
Never retries an uncertain CAD mutation; only creates a new import in this demo.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path.home()/'AppData/Local/SolidWorksMCP/versions/0.1.5'
sys.path.insert(0,str(RUNTIME/'examples/workflows'))
from mcp_stdio import MCPClient


def main():
    p = argparse.ArgumentParser();p.add_argument('--probe',action='store_true')
    args = p.parse_args()
    report = {}; destination = ROOT/'output/car/solidworks_validation.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    try:
        with MCPClient(RUNTIME/'solidworks-mcp.exe', ROOT) as client:
            def call(name, **kwargs):
                result = client.tool(name,kwargs)
                report[name] = result
                destination.write_text(json.dumps(report,indent=2),encoding='utf-8')
                print(name, json.dumps(result,ensure_ascii=False), flush=True)
                return result
            call('connect',launch=True,visible=True)
            call('observe_session')
            if args.probe: return
            opened = call('open_document',path='output/car/car.step')
            doc = opened['document']['document_id']
            call('show_document',document_id=doc,view='isometric',fit=True)
            inspection = call('inspect_surfaces',document_id=doc,checks=['faults','gaps','continuity'],samples=5,image=True)
            assert inspection['status']=='inspected'
            call('save_document',document_id=doc,path='output/car/car.sldprt')
            call('preview_document',document_id=doc,width=1400,height=900)
            call('close_document',document_id=doc)
            reopened=call('open_document',path='output/car/car.sldprt')
            doc=reopened['document']['document_id']
            call('inspect_surfaces',document_id=doc,checks=['faults','gaps'],image=False)
    except Exception as exc:
        report['error']=str(exc)
        destination.write_text(json.dumps(report,indent=2),encoding='utf-8')
        raise


if __name__=='__main__':main()
