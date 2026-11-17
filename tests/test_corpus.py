import tempfile
from pathlib import Path
import unittest

from vegas_subtitles.corpus import assign_groups, merge_spans, parse_project, unwrap_speaker_stars
from vegas_subtitles.metrics import word_errors


class CorpusTests(unittest.TestCase):
    def test_censor_stars_remain_after_speaker_unwrap(self):
        self.assertEqual(unwrap_speaker_stars("*  слово бл**ь *"), "слово бл**ь")

    def test_muted_source_pieces_merge_without_losing_uncensored_audio(self):
        pieces = [{"path":"recording.mkv", "stream":0, "rate":1,
                   "timeline_start":0., "timeline_end":1., "source_start":10., "source_end":11.},
                  {"path":"recording.mkv", "stream":0, "rate":1,
                   "timeline_start":1., "timeline_end":1.3, "source_start":11., "source_end":11.3},
                  {"path":"recording.mkv", "stream":0, "rate":1,
                   "timeline_start":1.3, "timeline_end":2., "source_start":11.3, "source_end":12.}]
        spans = merge_spans(pieces)
        self.assertEqual(len(spans), 1)
        self.assertEqual((spans[0]["source_start"], spans[0]["source_end"]), (10.,12.))

    def test_source_jump_is_not_falsely_merged(self):
        pieces = [{"path":"a", "stream":0, "rate":1, "timeline_start":0, "timeline_end":1,
                   "source_start":10, "source_end":11},
                  {"path":"a", "stream":0, "rate":1, "timeline_start":1, "timeline_end":2,
                   "source_start":30, "source_end":31}]
        self.assertEqual(len(merge_spans(pieces)), 2)

    def test_related_projects_never_leak_across_splits(self):
        rows = [{"project":p, "spans":[{"path":source}]} for p,source in
                [("original","a"),("remaster","A"),("second","b"),("third","c"),("fourth","d")]]
        assign_groups(rows)
        self.assertEqual(rows[0]["group"], rows[1]["group"])
        self.assertEqual(rows[0]["split"], rows[1]["split"])

    def test_bad_labels_and_overlaps_are_flagged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"project.xml"
            media = Path(directory)/"raw.mkv"
            media.touch()
            xml = f'''<project duration="2"><track index="0" type="Video">
              <event index="0" start="0" end="1"><generator speaker="self"><text>слово бл**ь</text></generator></event>
              <event index="1" start="1" end="2"><generator speaker="self"><text>translated English text</text></generator></event>
              <event index="2" start="0" end="2" path="{media}"/>
              </track><track index="1" type="Video"><event index="0" start=".5" end="1.2"><generator speaker="friend"><text>* другая речь *</text></generator></event></track>
              <track index="2" type="Audio"><event index="0" start="0" end="2" path="{media}" offset="5" rate="1" stream="0" gain="0"/></track></project>'''
            path.write_text(xml, encoding="utf-8")
            rows, _ = parse_project(path)
            self.assertIn("masked_profanity_label", rows[0]["issues"])
            self.assertIn("overlapping_speakers", rows[0]["issues"])
            self.assertIn("non_russian_label_needs_review", rows[1]["issues"])
            self.assertEqual(rows[0]["spans"][0]["source_start"], 5)

    def test_digit_substituted_cyrillic_label_requires_review(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"project.xml"
            path.write_text('''<project><track index="0" type="Video">
                <event index="0" start="0" end="2"><generator speaker="self">
                <text>4лен, типа, ну</text></generator></event></track></project>''', encoding="utf-8")
            rows, _ = parse_project(path)
            self.assertIn("obfuscated_label_needs_review", rows[0]["issues"])


class MetricTests(unittest.TestCase):
    def test_punctuation_case_and_yo_do_not_change_score(self):
        self.assertEqual(word_errors("Ещё один!", "еще один.")["errors"], 0)

    def test_missing_and_invented_words_are_counted(self):
        self.assertEqual(word_errors("два слова здесь", "два здесь еще")["errors"], 2)

    def test_empty_reference_is_explicit(self):
        self.assertIsNone(word_errors("", "несуществующая речь")["wer"])


if __name__ == "__main__":
    unittest.main()
