"""Bounded Poe REST requests, with full-content caching and no secret logging."""
from __future__ import annotations
import json
import os
from pathlib import Path
import time
import uuid
import httpx


class BudgetExceeded(RuntimeError):
    pass


class PoeClient:
    def __init__(self, directory, *, api_key=None, model='gpt-5.4-mini', max_calls=10):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key or os.environ.get('POE_API_KEY', '')
        self.model = model
        self.max_calls = max_calls
        self.log = self.directory / 'calls.jsonl'
        self.calls = len(self.log.read_text().splitlines()) if self.log.exists() else 0

    def ask(self, request_id, system, payload, temperature=0.2):
        if not request_id.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Unsafe request ID')
        request = {'model':self.model, 'messages':[{'role':'system','content':system},
                   {'role':'user','content':json.dumps(payload, ensure_ascii=False)}],
                   'temperature':temperature, 'max_tokens':7000, 'stream':False}
        path = self.directory / f'{request_id}.json'
        if path.exists():
            saved = json.loads(path.read_text())
            if saved['request'] != request:
                raise ValueError('Cached request changed; use a new protocol/output directory')
            return saved['parsed']
        if self.calls >= self.max_calls:
            raise BudgetExceeded(f'Actual API request budget exhausted: {self.calls}/{self.max_calls}')
        if not self.api_key:
            raise RuntimeError('POE_API_KEY is not set; run poe in the same interactive shell')
        self.calls += 1
        entry={'call_id':str(uuid.uuid4()),'request_id':request_id,'started_at':time.time(),'request':request}
        # Persist the attempted request BEFORE sending; crashed/time-out calls still count.
        with self.log.open('a') as f:
            f.write(json.dumps(entry,ensure_ascii=False)+'\n'); f.flush(); os.fsync(f.fileno())
        started=time.monotonic()
        try:
            with httpx.Client(timeout=httpx.Timeout(240, connect=30), follow_redirects=False) as client:
                response=client.post('https://api.poe.com/v1/chat/completions',
                    headers={'Authorization':f'Bearer {self.api_key}'},json=request)
                if response.status_code != 200:
                    raise RuntimeError(f'Poe HTTP status {response.status_code}')
                data=response.json()
        except httpx.HTTPError as exc:
            raise RuntimeError(f'Poe transport failed: {type(exc).__name__}') from None
        content=data['choices'][0]['message']['content']
        if not isinstance(content,str):
            raise ValueError('Poe response is not text')
        stripped=content.strip()
        if stripped.startswith('```'):
            stripped=stripped.split('\n',1)[1].rsplit('```',1)[0].strip()
        record={'request':request,'response_text':content,'response_model':data.get('model'),
                'usage':data.get('usage'),'elapsed_seconds':time.monotonic()-started,
                'finish_reason':data['choices'][0].get('finish_reason')}
        try:
            parsed=json.loads(stripped)
            if not isinstance(parsed,dict):raise ValueError('Expected JSON object')
        except (ValueError,TypeError):
            (self.directory/f'{request_id}.invalid.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
            raise ValueError('Agent returned invalid JSON object') from None
        record['parsed']=parsed
        temp=path.with_suffix('.tmp')
        temp.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
        temp.replace(path)
        return parsed
