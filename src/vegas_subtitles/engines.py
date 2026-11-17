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


class Parakeet:
    def __init__(self, folder: Path):
        import torch
        from transformers import AutoModelForTDT, AutoProcessor
        torch.set_num_threads(8)
        self.processor=AutoProcessor.from_pretrained(folder,local_files_only=True)
        self.model=AutoModelForTDT.from_pretrained(folder,dtype=torch.float32,
            device_map='cuda',attn_implementation='sdpa',local_files_only=True).eval()
        self.settings={'engine':'transformers-parakeet-tdt','dtype':'float32','decoding':'greedy'}

    def transcribe(self, audio: Path) -> str:
        import torch
        with torch.inference_mode():
            inputs=self.processor(read_wav(audio),sampling_rate=16000,return_tensors='pt')
            inputs=inputs.to(self.model.device,self.model.dtype)
            output=self.model.generate(**inputs,return_dict_in_generate=True)
            return self.processor.decode(output.sequences,skip_special_tokens=True)[0].strip()


class RussianWhisper:
    def __init__(self, folder: Path, num_beams: int=1):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        torch.set_num_threads(8)
        self.processor=WhisperProcessor.from_pretrained(folder,local_files_only=True)
        self.model=WhisperForConditionalGeneration.from_pretrained(folder,dtype=torch.bfloat16,
            device_map='cuda',attn_implementation='sdpa',local_files_only=True).eval()
        self.num_beams=num_beams
        self.settings={'engine':'transformers-whisper','dtype':'bfloat16','num_beams':num_beams,
                       'max_new_tokens':256,'language':'ru','task':'transcribe','vad':False}

    def transcribe(self, audio: Path) -> str:
        import torch
        with torch.inference_mode():
            inputs=self.processor(read_wav(audio),sampling_rate=16000,return_tensors='pt',return_attention_mask=True)
            output=self.model.generate(**inputs.to(self.model.device,self.model.dtype),
                language='ru',task='transcribe',num_beams=self.num_beams,do_sample=False,max_new_tokens=256)
            return self.processor.batch_decode(output,skip_special_tokens=True)[0].strip()


class TOne:
    def __init__(self, folder: Path):
        import torch  # Load the CUDA/cuDNN libraries supplied by the pinned base.
        import onnxruntime as ort
        import onnx_asr
        if 'CUDAExecutionProvider' not in ort.get_available_providers():
            raise RuntimeError('CUDA ONNX provider is unavailable')
        import numpy as np
        profile_dir=Path('local/runtime/onnx-profile');profile_dir.mkdir(parents=True,exist_ok=True)
        options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
        options.enable_profiling=True;options.profile_file_prefix=str(profile_dir/'tone')
        self.model=onnx_asr.load_model('t-tech/t-one',str(folder),providers=['CUDAExecutionProvider'],
            sess_options=options)
        session=self.model.asr._model
        if 'CUDAExecutionProvider' not in session.get_providers():
            raise RuntimeError('T-one fell back to CPU')
        self.model.recognize(np.zeros(16000,dtype=np.float32),sample_rate=16000)
        trace_path=session.end_profiling()
        trace=json.loads(Path(trace_path).read_text())
        used={e.get('args',{}).get('provider') for e in trace}
        if 'CUDAExecutionProvider' not in used:
            raise RuntimeError('T-one profiling did not record any CUDA kernels')
        self.settings={'engine':'onnx-asr-tone','decoding':'greedy','kenlm':False,
                       'requested_provider':'CUDAExecutionProvider','input_sample_rate':16000,'model_sample_rate':8000,
                       'cuda_kernels_verified':True,'profile_trace':trace_path,'warmup':'1 second synthetic silence'}

    def transcribe(self, audio: Path) -> str:
        return self.model.recognize(read_wav(audio),sample_rate=16000).strip()


def load_engine(kind: str, folder: Path):
    return {'whisper':Whisper,'qwen':Qwen,'gigaam':GigaAM,'parakeet':Parakeet,
            'russian-whisper':RussianWhisper,'russian-whisper-beam5':lambda p:RussianWhisper(p,num_beams=5),
            't-one':TOne}[kind](folder)
