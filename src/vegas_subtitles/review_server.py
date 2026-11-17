"""Loopback-only review UI. Original media and VEGAS projects are never writable here."""
from __future__ import annotations
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import re
import threading
from pathlib import Path
from urllib.parse import urlsplit

from .evaluation import fingerprint


def serve(dataset: Path, batch: Path, saved: Path, host: str='127.0.0.1', port: int=8765):
    package=json.loads(dataset.read_text(encoding='utf-8'))
    spec=json.loads(batch.read_text(encoding='utf-8'))
    ids=spec['ids']
    selected={r['id']:r for r in package['items'] if r['id'] in ids}
    if len(selected)!=len(ids):raise ValueError('Review batch contains missing or duplicate IDs')
    dataset_hash=fingerprint(dataset)
    save_lock=threading.Lock()
    predictions=spec.get('predictions',{})
    web=Path(__file__).resolve().parents[2]/'web'
    class Handler(BaseHTTPRequestHandler):
        def send(self,status,data,kind='application/json; charset=utf-8'):
            if not isinstance(data,bytes):data=json.dumps(data,ensure_ascii=False).encode('utf-8')
            self.send_response(status);self.send_header('Content-Type',kind)
            self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(data)

        def audio(self,file):
            data=file.read_bytes();total=len(data);start=0;end=total-1
            requested=self.headers.get('Range')
            if requested:
                match=re.fullmatch(r'bytes=(\d*)-(\d*)',requested.strip())
                if not match or not any(match.groups()):return self.send(416,{'error':'Invalid byte range'})
                lo,hi=match.groups()
                if lo:
                    start=int(lo);end=min(int(hi),total-1) if hi else total-1
                else:start=max(0,total-int(hi))
                if start>end or start>=total:return self.send(416,{'error':'Unsatisfiable byte range'})
            self.send_response(206 if requested else 200)
            self.send_header('Content-Type','audio/wav');self.send_header('Accept-Ranges','bytes')
            self.send_header('Content-Length',str(end-start+1));self.send_header('Cache-Control','no-store')
            if requested:self.send_header('Content-Range',f'bytes {start}-{end}/{total}')
            self.end_headers();self.wfile.write(data[start:end+1])

        def do_GET(self):
            path=urlsplit(self.path).path
            if path=='/':return self.send(200,(web/'review-v2.html').read_bytes(),'text/html; charset=utf-8')
            if path=='/review.js':return self.send(200,(web/'review.js').read_bytes(),'text/javascript; charset=utf-8')
            if path=='/review.css':return self.send(200,(web/'review.css').read_bytes(),'text/css; charset=utf-8')
            if path=='/api/data':
                decisions={}
                if saved.exists():
                    stored=json.loads(saved.read_text(encoding='utf-8'))
                    if stored['dataset_sha256']==dataset_hash:
                        decisions={r['id']:r for r in stored['decisions']}
                fields=('id','project','start','end','text','speaker','issues','duration','context_duration','trim_start','trim_end','split')
                rows=[{**{k:selected[i][k] for k in fields},'prediction':predictions.get(i,{}),'decision':decisions.get(i)} for i in ids]
                return self.send(200,{'title':spec.get('title','Проверка разметки'),'dataset_sha256':dataset_hash,'items':rows})
            if path.startswith(('/audio/','/clip/')) and path.endswith('.wav'):
                ident=path.rsplit('/',1)[-1][:-4]
                if ident not in selected:return self.send(404,{'error':'Unknown audio ID'})
                key='context_audio' if path.startswith('/audio/') else 'audio'
                file=dataset.parent/selected[ident][key]
                return self.audio(file)
            return self.send(404,{'error':'Not found'})

        def do_POST(self):
            if self.path!='/api/save':return self.send(404,{'error':'Not found'})
            expected='http://'+self.headers.get('Host','')
            if self.headers.get('Origin')!=expected:return self.send(403,{'error':'Invalid request origin'})
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=1024*1024:return self.send(413,{'error':'Invalid request size'})
            try:
                review=json.loads(self.rfile.read(length))
                if review['dataset_sha256']!=dataset_hash:raise ValueError('Dataset version changed')
                seen=set()
                for item in review['decisions']:
                    ident=item['id']
                    if ident not in selected or ident in seen:raise ValueError('Invalid review IDs')
                    seen.add(ident)
                    start,end=float(item['trim_start']),float(item['trim_end'])
                    if not math.isfinite(start+end) or not 0<=start<end<=selected[ident]['context_duration']+.0001:
                        raise ValueError('Invalid audio boundaries')
                    if item.get('training_approved'):
                        if not item.get('alignment_reviewed') or not item['text'].strip() or '*' in item['text']:
                            raise ValueError('Approved labels require spoken words and reviewed boundaries')
                        if item['speaker'] not in ('self','friend'):raise ValueError('Review speaker')
                        if review.get('schema',1)>=2 and (item.get('text_reviewed') is not True or item.get('coverage')!='full'):
                            raise ValueError('Training approval requires reviewed text and complete audio')
                saved.parent.mkdir(parents=True,exist_ok=True)
                with save_lock:
                    temporary=saved.with_suffix('.tmp')
                    temporary.write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
                    temporary.replace(saved)
                return self.send(200,{'saved':len(review['decisions'])})
            except (ValueError,KeyError,TypeError) as exc:return self.send(400,{'error':str(exc)})
    print(f'Review UI: http://{host}:{port}, {len(ids)} examples. Saving to {saved}',flush=True)
    ThreadingHTTPServer((host,port),Handler).serve_forever()
