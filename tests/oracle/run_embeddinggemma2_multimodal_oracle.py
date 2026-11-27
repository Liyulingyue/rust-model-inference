import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor

ROOT = Path('models/embeddinggemma-2')
OUT = Path('tests/oracle/embeddinggemma2_multimodal.json')
torch.set_num_threads(1)
model = AutoModel.from_pretrained(ROOT, local_files_only=True, dtype=torch.float32)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(ROOT, local_files_only=True)

# Same 48x48 RGB HWC input as the Rust smoke, then HF patchify order.
h = w = 48
pixels = np.array([(i * 0.001) % (2*np.pi) for i in range(h*w*3)], dtype=np.float32)
pixels = (np.sin(pixels) * 0.5 + 0.5).reshape(h, w, 3)
patches = pixels.reshape(h//16,16,w//16,16,3).transpose(0,2,1,3,4).reshape(-1,16*16*3)
xs, ys = np.meshgrid(np.arange(w//16), np.arange(h//16), indexing='xy')
pos = np.stack([xs, ys], axis=-1).reshape(-1,2)
with torch.no_grad():
    pv = torch.from_numpy(patches).unsqueeze(0)
    pp = torch.from_numpy(pos).unsqueeze(0)
    tower = model.vision_tower(pixel_values=pv, pixel_position_ids=pp, return_dict=True).last_hidden_state
    text = model.embed_vision(tower)
    image_ids = tokenizer(tokenizer.boi_token + tokenizer.image_token * (text.numel() // 512) + tokenizer.eoi_token, return_tensors='pt')['input_ids']
    image_embedding = model(input_ids=image_ids, pixel_values=pv, image_position_ids=pp).last_hidden_state.mean(dim=1)

# A deterministic PCM16 fixture. Keeping the samples integer-defined avoids
# platform-specific libm differences between Rust's `f32::sin` and NumPy.
sample_count=16000
pcm = np.array([(i * 37) % 20001 - 10000 for i in range(sample_count)], dtype=np.int16)
wave = pcm.astype(np.float32)/32768.0
extractor = Gemma4AudioFeatureExtractor()
feat = extractor([wave], return_tensors='pt')
with torch.no_grad():
    audio_out = model.audio_tower(feat['input_features'], feat['input_features_mask'], return_dict=True)
    audio_tower = audio_out.last_hidden_state
    audio_text = model.embed_audio(audio_tower)
    audio_ids = tokenizer(tokenizer.boa_token + tokenizer.audio_token * (audio_text.numel() // 512) + tokenizer.eoa_token, return_tensors='pt')['input_ids']
    audio_embedding = model(input_ids=audio_ids, input_features=feat['input_features'], input_features_mask=feat['input_features_mask']).last_hidden_state.mean(dim=1)

obj = {
  'vision_input_shape': list(pv.shape),
  'vision_pos': pos.tolist(),
  'vision_tower_shape': list(tower.shape),
  'vision_tower': tower[0].flatten().tolist(),
  'vision_text_shape': list(text.shape),
  'vision_text': text[0].flatten().tolist(),
  'vision_token_ids': image_ids[0].tolist(),
  'vision_embedding': image_embedding[0].tolist(),
  'audio_features_shape': list(feat['input_features'].shape),
  'audio_mask_shape': list(feat['input_features_mask'].shape),
  'audio_mask': feat['input_features_mask'][0].tolist(),
  'audio_tower_shape': list(audio_tower.shape),
  'audio_tower': audio_tower[0].flatten().tolist(),
  'audio_text_shape': list(audio_text.shape),
  'audio_text': audio_text[0].flatten().tolist(),
  'audio_token_ids': audio_ids[0].tolist(),
  'audio_embedding': audio_embedding[0].tolist(),
}
OUT.write_text(json.dumps(obj))
print('wrote', OUT, 'vision', tower.shape, text.shape, 'audio', audio_tower.shape, audio_text.shape)
