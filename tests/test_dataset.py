import json
from pathlib import Path
import tempfile
import unittest
import wave

from vegas_subtitles.dataset import export_reviewed
from vegas_subtitles.evaluation import fingerprint, from_reviewed
from vegas_subtitles.engines import native_giga_config
from vegas_subtitles.metrics import character_errors
from vegas_subtitles.training_text import gigaam_label
from vegas_subtitles.review_batch import prepare_batch


class ReviewedDatasetTests(unittest.TestCase):
    def fixture(self,root,**updates):
        wav=root/'context.wav'
        with wave.open(str(wav),'wb') as target:
            target.setnchannels(1);target.setsampwidth(2);target.setframerate(16000)
            target.writeframes(b'\x01\x00'*16000)
        dataset=root/'dataset.json'
        dataset.write_text(json.dumps({'items':[{'id':'one','split':'holdout','group':'recording-a',
            'context_audio':'context.wav','context_sha256':fingerprint(wav),'context_duration':1}]}))
        review=root/'review.json'
        decision={'id':'one','text':'привет','speaker':'self','trim_start':.2,'trim_end':.8,
                  'training_approved':True,'alignment_reviewed':True,**updates}
        review.write_text(json.dumps({'dataset_sha256':fingerprint(dataset),'decisions':[decision]}))
        return dataset,review

    def test_holdout_cannot_be_reassigned_by_review_and_audio_is_trimmed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);dataset,review=self.fixture(root,split='train')
            summary=export_reviewed(dataset,review,root/'out')
            self.assertEqual(summary['splits'],{'holdout':1})
            self.assertFalse((root/'out/train.jsonl').exists())
            with wave.open(str(root/'out/audio/one.wav'),'rb') as source:
                self.assertEqual(source.getnframes(),9600)

    def test_masked_labels_never_enter_training_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);dataset,review=self.fixture(root,text='бл**ь')
            with self.assertRaisesRegex(ValueError,'actual spoken words'):
                export_reviewed(dataset,review,root/'out')
            self.assertFalse((root/'out').exists())

    def test_approval_requires_explicit_boundary_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);dataset,review=self.fixture(root,alignment_reviewed=False)
            with self.assertRaisesRegex(ValueError,'alignment review'):
                export_reviewed(dataset,review,root/'out')

    def test_modified_audio_is_rejected_before_any_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);dataset,review=self.fixture(root)
            (root/'context.wav').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):
                export_reviewed(dataset,review,root/'out')
            self.assertFalse((root/'out').exists())

    def test_unchecked_examples_stay_out_of_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);dataset,review=self.fixture(root,training_approved=False)
            self.assertEqual(export_reviewed(dataset,review,root/'out')['approved'],0)

    def test_v2_approval_requires_the_separate_text_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);dataset,review=self.fixture(root,text_reviewed=False,coverage='full')
            payload=json.loads(review.read_text());payload['schema']=2;review.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError,'reviewed text'):
                export_reviewed(dataset,review,root/'out')

    def test_latin_label_is_preserved_but_excluded_from_russian_only_tsv(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);dataset,review=self.fixture(root,text='Я беру CZ')
            summary=export_reviewed(dataset,review,root/'out')
            self.assertEqual(summary['gigaam_excluded_labels'],1)
            record=json.loads((root/'out/holdout.jsonl').read_text(encoding='utf-8'))
            self.assertEqual(record['text'],'Я беру CZ')
            self.assertEqual(len((root/'out/holdout.tsv').read_text(encoding='utf-8').splitlines()),1)


class BackendConfigTests(unittest.TestCase):
    def test_development_diagnostic_refuses_holdout_examples(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest=root/'holdout.jsonl'
            manifest.write_text(json.dumps({'split':'holdout','label_source':'explicit_human_review','alignment_reviewed':True})+'\n')
            with self.assertRaisesRegex(ValueError,'requested split'):
                from_reviewed(manifest,root/'diagnostic','development')
            self.assertFalse((root/'diagnostic').exists())
    def test_review_queue_never_consumes_final_holdout_or_overwrites_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);dataset=root/'dataset.json'
            dataset.write_text(json.dumps({'items':[{'id':s,'split':s,'issues':[]} for s in ('train','validation','holdout','test')]}))
            output=root/'batch.json'
            prepare_batch(dataset,output,root/'audits',120)
            self.assertEqual(set(json.loads(output.read_text(encoding='utf-8'))['ids']),{'train','validation'})
            with self.assertRaisesRegex(ValueError,'already exists'):
                prepare_batch(dataset,output,root/'audits',120)
    def test_russian_training_normalization_keeps_spoken_words(self):
        self.assertEqual(gigaam_label('Ещё, бля!')['text'],'еще бля')
        self.assertTrue(gigaam_label('це зет')['supported'])

    def test_latin_and_digits_are_never_silently_removed_from_labels(self):
        self.assertFalse(gigaam_label('Я беру CZ')['supported'])
        self.assertFalse(gigaam_label('Дай 5 патронов')['supported'])
    def test_only_known_gigaam_architecture_classes_are_instantiated(self):
        self.assertEqual(native_giga_config({'_target_':'modeling_gigaam.CTCHead'}),
                         {'_target_':'gigaam.decoder.CTCHead'})
        with self.assertRaisesRegex(ValueError,'Unsupported'):
            native_giga_config({'_target_':'os.system'})

    def test_cer_preserves_spelling_errors_after_punctuation_normalization(self):
        self.assertEqual(character_errors('Ещё!','еще.')['character_errors'],0)
        self.assertEqual(character_errors('кот','код')['character_errors'],1)
