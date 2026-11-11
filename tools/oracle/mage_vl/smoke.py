"""Real CLI/HTTP/JEV and failure-boundary checks on the remote weight host."""
import argparse
import base64
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model')
    p.add_argument('vision')
    p.add_argument('image')
    p.add_argument('out',type=Path)
    p.add_argument('--vulkan', action='store_true', help='Require hardware Vulkan for CLI/HTTP/JEV')
    a = p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    binary = 'target/release-fast/rust-model-inference'
    base = [binary,'--model',a.model,'--threads','8','--max-context','2048','--prefill-batch-size','32']
    compute = ['--compute', 'vulkan' if a.vulkan else 'cpu']
    if a.vulkan:
        os.environ['RMI_COMPUTE_TRACE'] = '1'
    report = {}

    def cli(name,args,ok=True):
        command = base + args
        if '--compute' not in args:
            command += compute
        assert command.count('--compute') == 1, command
        result = subprocess.run(command,capture_output=True,text=True,timeout=900)
        (a.out/f'{name}.log').write_text(result.stdout+'\n'+result.stderr)
        assert (result.returncode==0)==ok,(name,result.stdout,result.stderr)
        report[name] = {'exit':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
        print(name,result.returncode,flush=True)
        return result.stdout

    generation = ['--prompt','Describe the colors briefly.','--max-tokens','8','--temperature','0']
    cpu = cli('cli-text-cpu',generation+['--compute','cpu'])
    assert cpu == cli('cli-text-auto',generation+['--compute','auto'])
    image_args = ['--mmproj',a.vision,'--image',a.image]
    image = cli('cli-image-cpu',generation+image_args+['--compute','cpu'])
    assert image == cli('cli-image-auto',generation+image_args+['--compute','auto'])
    cli('cli-image-f16',generation+image_args+['--kv-cache','f16'])
    assert cpu == cli('cli-mmproj-no-image',generation+['--mmproj',a.vision])
    cli('strict-vulkan',generation+['--compute','vulkan'],a.vulkan)
    if a.vulkan:
        assert 'scope=host_linear backend=Vulkan' in report['strict-vulkan']['stderr']
    cli('missing-vision',generation+['--image',a.image],False)
    cli('mismatched-vision',generation+['--image',a.image,'--mmproj',a.model],False)
    cli('context-before-vision',generation+image_args+['--max-context','2'],False)
    cli('output-budget-before-vision',generation+image_args+['--max-context','64','--max-tokens','64'],False)
    cli('video-rejected',generation+['--video',a.image],False)
    cli('dump-logits-rejected',generation+['--dump-logits'],False)
    question = {'text':'Are there colors?', 'options':['yes','no']}
    jev = ['--jev','--jev-context','Colors.','--jev-question',question['text'],
           '--jev-option','yes','--jev-option','no','--jev-output','json']
    text_jev = json.loads(cli('jev-text',jev))
    image_jev = json.loads(cli('jev-image',jev+image_args))
    cli('jev-context-before-vision',jev+image_args+['--max-context','2'],False)

    url = 'http://127.0.0.1:18083'
    def post(path,body,expected=200):
        request = Request(url+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        try:
            with urlopen(request,timeout=900) as response:
                status,value = response.status,json.load(response)
        except HTTPError as error:
            status,value = error.code,json.load(error)
        assert status==expected,(path,status,value)
        report[path+str(len(report))] = {'status':status,'body':value}
        (a.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(path,status,flush=True)
        return value

    with (a.out/'server.log').open('w') as log:
        server = subprocess.Popen(base+compute+['--mmproj',a.vision,'--serve','--host','127.0.0.1','--port','18083'],stdout=log,stderr=log)
        try:
            for _ in range(120):
                assert server.poll() is None,'server exited'
                try:
                    with urlopen(url+'/health',timeout=1):
                        break
                except (URLError,TimeoutError):
                    time.sleep(0.5)
            else:
                raise RuntimeError('server startup timeout')
            data = 'data:image/png;base64,'+base64.b64encode(Path(a.image).read_bytes()).decode()
            chat = {'messages':[{'role':'user','content':'Describe the colors briefly.'}],
                    'max_tokens':8,'temperature':0,'enable_thinking':True}
            response = post('/v1/chat/completions',chat)
            assert response['choices'][0]['message']['content'].strip() == cpu.strip(),response
            chat['messages'][0]['content'] = [{'type':'image_url','image_url':{'url':data}},
                                             {'type':'text','text':'Describe the colors briefly.'}]
            response = post('/v1/chat/completions',chat)
            visible = image.strip()
            if visible.startswith('<think>'):
                visible = visible.split('</think>',1)[-1].strip() if '</think>' in visible else ''
            assert response['choices'][0]['message']['content'].strip() == visible,response
            parts = [{'type':'text','text':'Describe the '},
                     {'type':'image_url','image_url':{'url':data}},
                     {'type':'text','text':'colors briefly.'}]
            interleaved = post('/v1/chat/completions',dict(chat,messages=[{'role':'user','content':parts}]))
            expected = interleaved['choices'][0]['message']['content'].strip()
            parts[1] = {'type':'image','source':{'type':'base64','media_type':'image/png','data':data.split(',',1)[1]}}
            anthropic = post('/v1/messages',dict(chat,messages=[{'role':'user','content':parts}]))
            assert ''.join(p.get('text','') for p in anthropic['content']).strip() == expected,anthropic
            parts[1] = {'type':'input_image','image_url':data}
            responses = post('/v1/responses',{'input':[{'role':'user','content':parts}],
                             'max_output_tokens':8,'temperature':0,'enable_thinking':True,'store':False})
            actual = ''.join(p.get('text','') for item in responses['output'] for p in item.get('content',[]))
            assert actual.strip() == expected,responses
            chat['messages'] += [{'role':'assistant','content':'Many colors.'},
                                 {'role':'user','content':'Name one color.'}]
            post('/v1/chat/completions',chat)
            chat['tools'] = [{'type':'function','function':{'name':'inspect_color','parameters':{'type':'object','properties':{}}}}]
            chat['tool_choice'] = 'auto'
            post('/v1/chat/completions',chat)
            history = chat['messages'] + [
                {'role':'assistant','content':'','tool_calls':[{'id':'color_1','type':'function',
                 'function':{'name':'inspect_color','arguments':'{}'}}]},
                {'role':'tool','tool_call_id':'color_1','content':'Red.'}]
            post('/v1/chat/completions',dict(chat,messages=history))
            post('/v1/chat/completions',{'messages':[{'role':'tool','content':'bad'}]},400)
            post('/v1/chat/completions',{'messages':[{'role':'user','content':'<|image_pad|>'}]},400)
            post('/v1/chat/completions',dict(chat,max_tokens=2048),400)
            for path,expected in [('/v1/jev/score',text_jev),('/v1/jev/image',image_jev)]:
                body = {'context':'Colors.','questions':[question]}
                if path.endswith('image'):
                    body['image_url'] = data
                value = post(path,body)
                actual,expect = value['results'][0]['probabilities'],expected['probabilities']
                assert actual.keys()==expect.keys()
                assert max(abs(actual[k]-expect[k]) for k in actual)<1e-6,(path,actual,expect)
        finally:
            server.terminate()
            server.wait(timeout=30)
    (a.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
