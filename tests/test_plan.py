import unittest
from vegas_subtitles.plan import validate_plan


class PlanTests(unittest.TestCase):
    def test_two_people_may_speak_at_the_same_time(self):
        validate_plan([{"start":1.,"end":2.,"speaker":"self","text":"один"},
                       {"start":1.,"end":2.,"speaker":"friend","text":"другой"}], [])

    def test_same_person_does_not_get_stacked_overlapping_captions(self):
        with self.assertRaisesRegex(ValueError,"Overlapping"):
            validate_plan([{"start":1.,"end":2.,"speaker":"self","text":"один"},
                           {"start":1.5,"end":2.5,"speaker":"self","text":"два"}], [])

    def test_unknown_voice_must_be_reviewed(self):
        with self.assertRaisesRegex(ValueError,"reviewed"):
            validate_plan([{"start":1.,"end":2.,"speaker":"unknown","text":"слово"}], [])
