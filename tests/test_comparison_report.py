import json
from pathlib import Path
import tempfile
import unittest
from vegas_subtitles.comparison_report import generate


class ComparisonReportTests(unittest.TestCase):
    def write_run(self,folder,set_hash,failures=None):
        folder.mkdir(parents=True)
        (folder/'summary.json').write_text(json.dumps({'failed_samples':failures or [],
            'examples':1,'audio_ids':['one'],'set_sha256':set_hash,'reference_words':1}))
        (folder/'samples.json').write_text(json.dumps([{'id':'one','reference':'привет','hypothesis':'привет','errors':0}]))

    def test_runs_from_different_audio_sets_cannot_be_ranked_together(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.write_run(root/'runs/a','a'*64);self.write_run(root/'runs/b','b'*64)
            with self.assertRaisesRegex(ValueError,'same frozen audio set'):
                generate(root/'runs',root/'report')
            self.assertFalse((root/'report').exists())

    def test_failed_run_cannot_silently_receive_a_good_score(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.write_run(root/'runs/a','a'*64,[{'id':'missing','error':'failed'}])
            with self.assertRaisesRegex(ValueError,'Incomplete run'):
                generate(root/'runs',root/'report')
