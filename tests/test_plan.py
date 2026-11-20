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

    def test_whisper_credit_blocks_export_even_inside_a_caption(self):
        for text in ("Субтитры сделал DimaTorzok", "Продолжаем. СУБТИТРЫ\nсоздавал dimatorzok!"):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, r"1\.250.*Whisper"):
                validate_plan([{"start":1.25,"end":2.,"speaker":"self","text":text}], [])

    def test_name_alone_does_not_block_export(self):
        validate_plan([{"start":1.,"end":2.,"speaker":"self","text":"Привет, DimaTorzok!"}], [])
