import unittest
from vegas_subtitles.profanity_metrics import diagnose,family


class ProfanityDiagnosticTests(unittest.TestCase):
    def test_benign_eb_and_huya_substrings_are_not_matched(self):
        for word in ('тебе','ребята','небо','страхуя','худой'):
            self.assertIsNone(family(word),word)

    def test_spelling_variant_is_matched_but_missing_repeat_is_counted(self):
        result=diagnose([{'reference':'блять, блять','hypothesis':'блядь'}])
        self.assertEqual(result['reference_occurrences'],2)
        self.assertEqual(result['matched_families'],1)
        self.assertEqual(result['recall'],.5)

    def test_extra_swear_on_clean_speech_is_reported(self):
        result=diagnose([{'reference':'спасибо тебе','hypothesis':'спасибо, сука'}])
        self.assertEqual(result['reference_occurrences'],0)
        self.assertEqual(result['extra_occurrences'],1)
        self.assertEqual(result['precision'],0)
