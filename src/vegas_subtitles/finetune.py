"""Adapt native GigaAM v3 RNN-T locally using verified speech labels."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import time

from .audio import read_wav
from .corpus import load_rows
from .engines import GigaAM, load_native_gigaam
from .evaluation import fingerprint
from .metrics import character_errors, word_errors


def reviewed_rows(path: Path, split: str) -> dict:
    rows = {}
    for row in load_rows(path):
        if (row['id'] in rows or row['split'] != split
                or row['label_source'] != 'explicit_human_review' or row['alignment_reviewed'] is not True):
            raise ValueError(f'Invalid or duplicate reviewed row: {row["id"]}')
        rows[row['id']] = row
    if not rows:
        raise ValueError(f'Empty {split} manifest')
    return rows


def verify_inputs(dataset: Path, validation_set: Path) -> tuple[list, list]:
    train = reviewed_rows(dataset / 'train.jsonl', 'train')
    validation = reviewed_rows(dataset / 'validation.jsonl', 'validation')
    for field in ('id', 'group', 'audio_sha256'):
        if {row[field] for row in train.values()} & {row[field] for row in validation.values()}:
            raise ValueError(f'Train/validation overlap in {field}')

    with (dataset / 'train.tsv').open(encoding='utf-8', newline='') as stream:
        tsv = list(csv.DictReader(stream, delimiter='\t'))
    usable = [row for row in train.values() if row['gigaam_normalization']['supported']]
    expected = [{'path': row['audio'], 'duration': str(row['duration']),
                 'transcription': row['gigaam_normalization']['text']} for row in usable]
    if not expected or tsv != expected:
        raise ValueError('Training TSV does not match the reviewed train manifest')

    package = json.loads(validation_set.read_bytes())
    items = package['items']
    if (package['split'] != 'validation' or package['reference_status'] != 'human_reviewed_text_and_boundaries'
            or package['source_manifest_sha256'] != fingerprint(dataset / 'validation.jsonl')
            or len(items) != len(validation) or {row['id'] for row in items} != validation.keys()):
        raise ValueError('Validation set does not match the frozen reviewed manifest')
    for row in items:
        if row != validation[row['id']]:
            raise ValueError(f'Validation metadata differs: {row["id"]}')

    for parent, rows in ((dataset, usable), (validation_set.parent, items)):
        for row in rows:
            audio = (parent / row['audio']).resolve()
            if not audio.is_relative_to(parent.resolve()) or fingerprint(audio) != row['audio_sha256']:
                raise ValueError(f'Audio path or checksum mismatch: {row["id"]}')
    return usable, items


def validation_score(engine: GigaAM, rows: list, folder: Path) -> dict:
    errors = words = chars = reference_chars = 0
    for row in rows:
        hypothesis = engine.transcribe(folder / row['audio'])
        word = word_errors(row['text'], hypothesis)
        char = character_errors(row['text'], hypothesis)
        errors += word['errors']
        words += word['reference_words']
        chars += char['character_errors']
        reference_chars += char['reference_characters']
    return dict(word_errors=errors, reference_words=words, wer=errors / words,
                character_errors=chars, reference_characters=reference_chars, cer=chars / reference_chars)


def rnnt_training_loss(model, wavs, wav_lengths, tokens, token_lengths):
    import torch
    from torchaudio.functional import rnnt_loss

    with torch.no_grad():
        features, lengths = model.preprocessor(wavs, wav_lengths)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        encoded, encoded_lengths = model.encoder(features, lengths)
    decoder, joint = model.head.decoder, model.head.joint
    embedded = decoder.embed(tokens)
    bos = embedded.new_zeros((tokens.size(0), 1, decoder.pred_hidden))
    predicted, _ = decoder.lstm(torch.cat((bos, embedded), dim=1).transpose(0, 1))
    acoustic = joint.enc(encoded.float().transpose(1, 2)).unsqueeze(2)
    language = joint.pred(predicted.transpose(0, 1)).unsqueeze(1)
    logits = joint.joint_net(acoustic + language)
    return rnnt_loss(logits, tokens.int(), encoded_lengths.int(), token_lengths.int(),
                     blank=decoder.blank_id, reduction='mean')


def train(args) -> None:
    import torch
    from torch.nn.utils import clip_grad_norm_
    from torch.nn.utils.rnn import pad_sequence

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA and BF16 are required for this training configuration')
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    rows, validation = verify_inputs(args.dataset, args.validation_set)
    model = load_native_gigaam(args.model).to('cuda').eval()
    if model.cfg.model_name != 'v3_rnnt' or not 1 <= args.encoder_layers <= len(model.encoder.layers):
        raise ValueError('Expected GigaAM v3 RNN-T and a valid number of encoder layers')
    blocks = model.encoder.layers[-args.encoder_layers:]
    model.requires_grad_(False)
    blocks.requires_grad_(True).train()
    model.head.requires_grad_(True).train()
    params = [param for param in model.parameters() if param.requires_grad]
    alphabet = {char: i for i, char in enumerate(model.cfg.decoding.vocabulary)}
    train_pairs = [(torch.from_numpy(read_wav(args.dataset / row['audio'])),
                    torch.tensor([alphabet[char] for char in row['gigaam_normalization']['text']], dtype=torch.long))
                   for row in rows]
    blank = model.head.decoder.blank_id
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01, fused=True)
    steps = args.epochs * ((len(rows) + args.batch_size - 1) // args.batch_size)
    warmup = max(1, min(10, steps // 10))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / warmup, 1.0) * (1 - 0.5 * step / steps))

    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.output / 'best-model'
    checkpoint.mkdir()
    shutil.copyfile(args.model / 'config.json', checkpoint / 'config.json')
    engine = GigaAM(args.model)
    run = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
           'model_sha256': {name: fingerprint(args.model / name) for name in ('config.json', 'pytorch_model.bin')},
           'train_manifest_sha256': fingerprint(args.dataset / 'train.tsv'),
           'validation_set_sha256': fingerprint(args.validation_set), 'train_examples': len(rows),
           'validation_examples': len(validation), 'trainable_parameters': sum(p.numel() for p in params),
           'arguments': {key: str(val) if isinstance(val, Path) else val for key, val in vars(args).items()},
           'encoder_autocast': 'bfloat16', 'epochs': []}
    print(json.dumps({key: val for key, val in run.items() if key != 'model_sha256'}), flush=True)
    generator = torch.Generator().manual_seed(args.seed)
    best = (float('inf'), float('inf'))
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        loss_sum = 0.0
        order = torch.randperm(len(train_pairs), generator=generator).tolist()
        for start in range(0, len(order), args.batch_size):
            subset = [train_pairs[i] for i in order[start:start + args.batch_size]]
            wavs, tokens = zip(*subset, strict=True)
            wav_lengths = torch.tensor([len(wav) for wav in wavs], device='cuda')
            token_lengths = torch.tensor([len(seq) for seq in tokens], device='cuda')
            padded_wavs = pad_sequence(wavs, batch_first=True).to('cuda')
            padded_tokens = pad_sequence(tokens, batch_first=True, padding_value=blank).to('cuda')
            optimizer.zero_grad(set_to_none=True)
            loss = rnnt_training_loss(model, padded_wavs, wav_lengths, padded_tokens, token_lengths)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite RNN-T loss at epoch {epoch}, example {start}')
            loss.backward()
            clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss.detach()) * len(subset)
            if start == 0 or (start // args.batch_size + 1) % 20 == 0:
                print(json.dumps(dict(epoch=epoch, trained_examples=start + len(subset),
                                      loss=loss_sum / (start + len(subset)))), flush=True)
        engine.model.load_state_dict(model.state_dict(), strict=True)
        score = validation_score(engine, validation, args.validation_set.parent)
        record = dict(epoch=epoch, train_loss=loss_sum / len(order), **score)
        run['epochs'].append(record)
        ranking = (score['word_errors'], score['character_errors'])
        if ranking < best:
            best = ranking
            run['best_epoch'] = epoch
            temporary = checkpoint / 'weights.tmp'
            torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, temporary)
            temporary.replace(checkpoint / 'pytorch_model.bin')
        run['torch_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        run['elapsed_seconds'] = time.monotonic() - started
        (args.output / 'training.json').write_text(json.dumps(run, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(record), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'dataset', 'validation-set', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--encoder-layers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not 0 < args.lr < 1:
        parser.error('Expected positive epochs, batch size and learning rate below one')
    train(args)


if __name__ == '__main__':
    main()
