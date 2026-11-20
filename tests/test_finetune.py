import json
from pathlib import Path
import tempfile
import unittest

from vegas_subtitles.evaluation import fingerprint
from vegas_subtitles.finetune import verify_inputs


class FineTuneInputTests(unittest.TestCase):
    def test_different_clip_ids_from_same_recording_cannot_cross_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ('train', 'validation'):
                row = {'id': split, 'group': 'same-recording', 'split': split,
                       'audio_sha256': split, 'label_source': 'explicit_human_review',
                       'alignment_reviewed': True}
                (root / f'{split}.jsonl').write_text(json.dumps(row) + '\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'overlap in group'):
                verify_inputs(root, root / 'set.json')

    def test_frozen_validation_rejects_edited_reference_even_with_same_ids_and_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = {'id': 'train', 'group': 'train-recording', 'split': 'train', 'audio_sha256': 'train-hash',
                     'label_source': 'explicit_human_review', 'alignment_reviewed': True,
                     'duration': 1.0, 'audio': 'train.wav',
                     'gigaam_normalization': {'supported': True, 'text': 'привет'}}
            val = {**train, 'id': 'validation', 'group': 'validation-recording', 'split': 'validation',
                   'audio_sha256': 'validation-hash', 'audio': 'validation.wav', 'text': 'пока'}
            for row in (train, val):
                (root / f'{row["split"]}.jsonl').write_text(json.dumps(row) + '\n', encoding='utf-8')
            (root / 'train.tsv').write_text('path\tduration\ttranscription\ntrain.wav\t1.0\tпривет\n', encoding='utf-8')
            frozen = {'split': 'validation', 'reference_status': 'human_reviewed_text_and_boundaries',
                      'source_manifest_sha256': fingerprint(root / 'validation.jsonl'),
                      'items': [{**val, 'text': 'другая реплика'}]}
            target = root / 'set.json'
            target.write_text(json.dumps(frozen), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Validation metadata differs'):
                verify_inputs(root, target)
