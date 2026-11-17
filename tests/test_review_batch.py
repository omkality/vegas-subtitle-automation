import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

from vegas_subtitles.evaluation import fingerprint
from vegas_subtitles.review_batch import extend_context, prepare_batch


class NextBatchTests(unittest.TestCase):
    def test_previous_queue_ids_are_excluded_even_when_not_approved(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);dataset=root/'dataset.json'
            dataset.write_text(json.dumps({'items':[{'id':i,'split':s,'issues':[],'group':i,
                'speaker':'self','duration':1} for i,s in [('old','train'),('new','train'),('val','validation'),('held','holdout')]]}))
            previous=root/'old.json';previous.write_text(json.dumps({'ids':['old']}))
            output=root/'next.json'
            prepare_batch(dataset,output,root/'audit',100,(previous,),'balanced')
            result=json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(set(result['ids']),{'new','val'})
            self.assertEqual(result['excluded_batches'][0]['sha256'],fingerprint(previous))

    def test_exclusions_from_another_dataset_fail_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);dataset=root/'dataset.json';dataset.write_text('{"items":[]}')
            previous=root/'old.json';previous.write_text('{"ids":[],"dataset_sha256":"different"}')
            with self.assertRaisesRegex(ValueError,'different dataset'):
                prepare_batch(dataset,root/'next.json',root/'audit',100,(previous,))
            self.assertFalse((root/'next.json').exists())

    def fixture(self,root,split='train'):
        audio=root/'clip.wav'
        self.write_audio(audio,1)
        row={'id':'one','split':split,'audio':'clip.wav','audio_sha256':fingerprint(audio),
            'duration':1,'spans':[{'rate':1,'source_start':.4,'source_end':1.4}],
            'training_approved':False,'alignment_reviewed':False,'context_audio':'clip.wav'}
        dataset=root/'dataset.json';dataset.write_text(json.dumps({'items':[row]}))
        batch=root/'batch.json';batch.write_text(json.dumps({'ids':['one'],'dataset_sha256':fingerprint(dataset)}))
        return dataset,batch

    @staticmethod
    def write_audio(path,duration):
        path.parent.mkdir(parents=True,exist_ok=True)
        with wave.open(str(path),'wb') as w:
            w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(b'\1\0'*round(duration*16000))

    def test_wider_context_preserves_base_audio_split_and_original_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);dataset,batch=self.fixture(root)
            old_bytes=dataset.read_bytes();old_audio=(root/'clip.wav').read_bytes();seen=[]
            def extract(row,destination):
                span=row['spans'][0];seen.append(span)
                self.write_audio(destination,span['source_end']-span['source_start'])
            with patch('vegas_subtitles.audio.extract_row',side_effect=extract):
                extend_context(dataset,batch,root/'extended',2.5)
            row=json.loads((root/'extended/dataset.json').read_text(encoding='utf-8'))['items'][0]
            self.assertEqual(seen[0]['source_start'],0)
            self.assertAlmostEqual(seen[0]['source_end'],3.9)
            self.assertAlmostEqual(row['trim_start'],.4)
            self.assertAlmostEqual(row['trim_end'],1.4)
            self.assertEqual(row['split'],'train')
            self.assertFalse(row['training_approved'])
            self.assertEqual((root/'extended'/row['audio']).read_bytes(),old_audio)
            self.assertEqual(dataset.read_bytes(),old_bytes)
            self.assertEqual(json.loads((root/'extended/batch.json').read_text(encoding='utf-8'))['dataset_sha256'],fingerprint(root/'extended/dataset.json'))

    def test_holdout_is_never_extracted_for_the_review_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);dataset,batch=self.fixture(root,'holdout')
            with patch('vegas_subtitles.audio.extract_row') as extract:
                with self.assertRaisesRegex(ValueError,'excludes test and holdout'):
                    extend_context(dataset,batch,root/'extended')
                extract.assert_not_called()
            self.assertFalse((root/'extended').exists())
