"""F32 CPU reference checkpoints from Transformers + pinned Tencent prompts."""
import argparse,json,sys
from pathlib import Path
import torch,numpy as np
import transformers
from PIL import Image,ImageDraw
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from tools.converter.wevisone_vl.convert_wevisone_vl import ORACLE_COMMIT
import subprocess

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'],text=True).strip()==ORACLE_COMMIT
    assert not subprocess.check_output(['git','-C',str(a.source),'status','--porcelain'])
    sys.path.insert(0,str(a.source.resolve()));from wevisone_vl.prompts import build_prompt
    a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(8)
    image=Image.new('RGB',(128,128),'white');ImageDraw.Draw(image).rectangle((32,24,96,104),fill='red');image.save(a.out/'image.png')
    processor=AutoProcessor.from_pretrained(a.model,min_pixels=16384,max_pixels=2073600)
    instruction=build_prompt('segmentation','the red rectangle')
    messages=[{'role':'system','content':'You are a helpful assistant.'},{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':instruction}]}]
    template=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    inputs=processor(text=[template],images=[image],return_tensors='pt')
    info={'instruction':instruction,'prompt':template,'token_ids':inputs.input_ids[0].tolist(),'grid_thw':inputs.image_grid_thw.tolist(),'tokenizer_samples':[],'torch':torch.__version__,'transformers':transformers.__version__}
    for text in ['你好，WeVisOne!\n\t  x','<box><0><2><999><1000><OBJ></box>','👩‍💻 café １２３']:
        info['tokenizer_samples'].append({'text':text,'ids':processor.tokenizer.encode(text,add_special_tokens=False)})
    model=Qwen3VLForConditionalGeneration.from_pretrained(a.model,dtype=torch.float32,attn_implementation='eager').eval()
    def save(name,t): (a.out/f'{name}.f32').write_bytes(t.detach().float().cpu().contiguous().numpy().tobytes())
    hooks=[]
    for i in [5,11,17,23]: hooks.append(model.model.visual.blocks[i].register_forward_hook(lambda m,i_,o,idx=i:save(f'vit{idx}',o[0] if isinstance(o,tuple) else o)))
    hooks.append(model.model.language_model.norm.register_forward_pre_hook(lambda m,i:save('raw',i[0])))
    with torch.inference_mode():
        positions,delta=model.model.get_rope_index(inputs.input_ids,inputs.mm_token_type_ids,inputs.image_grid_thw,attention_mask=inputs.attention_mask)
        info['positions']=positions[:,0,:].T.tolist()
        out=model(**inputs,use_cache=False)
        save('logits',out.logits[0,-1]);save('pixel_values',inputs.pixel_values)
        vision=model.model.get_image_features(inputs.pixel_values,inputs.image_grid_thw)
        save('projected',torch.cat(vision.pooler_output));save('deepstack',torch.stack([torch.cat(level) for level in vision.deepstack_features]))
        for hook in hooks:hook.remove()
        generated=model.generate(**inputs,max_new_tokens=32,do_sample=False,repetition_penalty=1.05,use_cache=True)
        info['greedy_ids']=generated[0,inputs.input_ids.shape[1]:].tolist()
        info['response']=processor.tokenizer.decode(info['greedy_ids'],skip_special_tokens=False)
        from wevisone_vl.engine import parse_seg_boxes
        from wevisone_vl.config import Qwen3VLSegConfig
        from wevisone_vl.mask_decoder import WeVisOneMaskDecoder
        from wevisone_vl.feature_bridge import VisualFeatureCapture,LastHiddenCapture,per_sample_premerge_feature_maps
        response=info['response'].split('<|im_end|>')[0]
        teacher=processor(text=[processor.apply_chat_template(messages+[{'role':'assistant','content':response}],tokenize=False,add_generation_prompt=False)],images=[image],return_tensors='pt')
        with VisualFeatureCapture(model,feature_layers=(5,11,17,23)) as visual,LastHiddenCapture(model) as hidden:
            model(**teacher,use_cache=False)
        obj=hidden.hidden[teacher.input_ids==152678]
        save('obj_hidden',obj)
        maps=per_sample_premerge_feature_maps(visual.premerge_features,(5,11,17,23),teacher.image_grid_thw,2)[0]
        sidecar=torch.load(a.model/'mask_decoder.pt',map_location='cpu',weights_only=True)
        decoder=WeVisOneMaskDecoder([1024]*4,2560,Qwen3VLSegConfig.from_dict(sidecar['cfg'])).eval();decoder.load_state_dict(sidecar['state_dict'],strict=True)
        canvas=torch.from_numpy(np.asarray(image.resize((1024,1024),Image.Resampling.BICUBIC)).copy()).permute(2,0,1).float()[None]/255
        boxes=torch.tensor(parse_seg_boxes(response))
        masks=decoder(canvas,maps,obj,boxes,(8,8))['mask_logits']
        save('full_mask_logits',masks)
        alpha=torch.nn.functional.interpolate(masks.float().sigmoid()[None],size=(128,128),mode='bilinear',align_corners=False)[0]
        save('alpha',alpha)
        info['teacher_ids']=teacher.input_ids[0].tolist()
        info['boxes']=boxes.tolist()
    (a.out/'fixture.json').write_text(json.dumps(info,ensure_ascii=False,indent=2))
    (a.out/'request.json').write_text(json.dumps({'image':str((a.out/'image.png').resolve()),'task':'segmentation','query':'the red rectangle','max_new_tokens':32}))
    print(info['response'],flush=True)
if __name__=='__main__': main()
