"""Local, independently loaded ASR backends for paired evaluation."""
from __future__ import annotations

import json
from pathlib import Path

from .audio import read_wav


class Whisper:
    def __init__(self, folder: Path):
        from faster_whisper import WhisperModel
        self.model=WhisperModel(str(folder),device='cuda',compute_type='float16',cpu_threads=4)
        self.settings={'engine':'faster-whisper','dtype':'float16','beam_size':5,'vad':True}

    def transcribe(self, audio: Path) -> str:
        segments,_=self.model.transcribe(read_wav(audio),language='ru',beam_size=5,
            vad_filter=True,condition_on_previous_text=False,word_timestamps=True)
        return ' '.join(s.text.strip() for s in segments)


class Qwen:
    def __init__(self, folder: Path):
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor
        torch.set_num_threads(8)
        self.processor=AutoProcessor.from_pretrained(folder,local_files_only=True)
        self.model=AutoModelForMultimodalLM.from_pretrained(folder,dtype=torch.bfloat16,
            device_map='cuda',attn_implementation='sdpa',local_files_only=True).eval()
        self.settings={'engine':'transformers','dtype':'bfloat16','decoding':'greedy','max_new_tokens':256,'language':'Russian'}

    def transcribe(self, audio: Path) -> str:
        import torch
        with torch.inference_mode():
            inputs=self.processor.apply_transcription_request(audio=read_wav(audio),language='Russian')
            inputs=inputs.to(self.model.device,self.model.dtype)
            output=self.model.generate(**inputs,max_new_tokens=256,do_sample=False)
            generated=output[:,inputs['input_ids'].shape[1]:]
            return self.processor.decode(generated,return_format='transcription_only')[0].strip()


GIGA_TARGETS={
    'FeatureExtractor':'gigaam.preprocess.FeatureExtractor',
    'ConformerEncoder':'gigaam.encoder.ConformerEncoder',
    'RNNTHead':'gigaam.decoder.RNNTHead',
    'CTCHead':'gigaam.decoder.CTCHead',
    'RNNTGreedyDecoding':'gigaam.decoding.RNNTGreedyDecoding',
    'CTCGreedyDecoding':'gigaam.decoding.CTCGreedyDecoding',
}


def native_giga_config(value):
    """Map the publisher's HF architecture config to its pinned native implementation."""
    if isinstance(value,list):return [native_giga_config(v) for v in value]
    if not isinstance(value,dict):return value
    result={k:native_giga_config(v) for k,v in value.items()}
    if '_target_' in result:
        name=result['_target_'].rsplit('.',1)[-1]
        if name not in GIGA_TARGETS:raise ValueError(f'Unsupported GigaAM class {name}')
        result['_target_']=GIGA_TARGETS[name]
    return result


class GigaAM:
    def __init__(self, folder: Path):
        import torch
        from omegaconf import OmegaConf
        from gigaam.model import GigaAMASR
        torch.set_num_threads(8)
        data=json.loads((folder/'config.json').read_text())
        cfg=native_giga_config(data['cfg']['model']['cfg'])
        if cfg.get('decoding',{}).get('model_path'):
            cfg['decoding']['model_path']=str(folder/'tokenizer.model')
        self.model=GigaAMASR(OmegaConf.create(cfg))
        state=torch.load(folder/'pytorch_model.bin',map_location='cpu',weights_only=True)
        if all(k.startswith('model.') for k in state):
            state={k.removeprefix('model.'):v for k,v in state.items()}
        self.model.load_state_dict(state,strict=True)
        self.model=self.model.eval().to('cuda')
        self.model.encoder=self.model.encoder.half()
        self.settings={'engine':'native-gigaam','encoder_dtype':'float16','decoding':'greedy',
                       'vendor_revision':'7447938d791c4f3e643386ee22c33777004293a5','strict_weight_load':True}

    def transcribe(self, audio: Path) -> str:
        return self.model.transcribe(str(audio)).text.strip()


def load_engine(kind: str, folder: Path):
    return {'whisper':Whisper,'qwen':Qwen,'gigaam':GigaAM}[kind](folder)
