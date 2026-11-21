"""Provider-neutral HTTP JSON bridge; callers supply the model service explicitly.

POST {schema, model, observation}; response {decision, usage?}. API credentials
are supplied separately and never included in observations, histories or errors.
"""
import json
from urllib import error, request


class HttpJsonProvider:
    def __init__(self,url,model,*,token=None,timeout=30.,max_response_bytes=1<<20):
        from urllib.parse import urlsplit
        parsed=urlsplit(url)
        if parsed.scheme not in ('http','https') or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError('Expected explicit HTTP(S) model service URL without embedded credentials')
        if not 0<timeout<=60:raise ValueError('Provider timeout must be within 60 seconds')
        self.url=url;self.model=model;self._token=token;self.timeout=timeout
        self.max_response_bytes=max_response_bytes;self.calls=0;self.last_usage=None

    def __call__(self,observation):
        payload=json.dumps({'schema':'cad-model-request-v1','model':self.model,'observation':observation},allow_nan=False).encode()
        headers={'Content-Type':'application/json','Accept':'application/json'}
        if self._token:headers['Authorization']='Bearer '+self._token
        req=request.Request(self.url,data=payload,headers=headers,method='POST')
        self.calls+=1
        try:
            with request.urlopen(req,timeout=self.timeout) as response:
                raw=response.read(self.max_response_bytes+1)
        except error.HTTPError as e:raise ValueError(f'Model service HTTP status {e.code}') from None
        except error.URLError:raise ValueError('Model service connection failed') from None
        except TimeoutError:raise ValueError('Model service timeout') from None
        if len(raw)>self.max_response_bytes:raise ValueError('Model response exceeded size limit')
        result=json.loads(raw)
        if not isinstance(result,dict) or 'decision' not in result:raise ValueError('Model service must return a decision object')
        self.last_usage=result.get('usage')
        return result['decision']
