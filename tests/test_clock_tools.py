"""Synthetic cases only; no project outputs or labels are consulted."""
import hashlib
import json
import subprocess
import sys
import unittest

from pipeline import clock_check
from pipeline.clock_check import check
from pipeline.clock_normalize import normalize_clock


class ClockCheckTests(unittest.TestCase):
    def test_synthetic_cases(self):
        cases = [
            ("О 21:05.", "At 9:05 p.m.", "match"),
            ("О 00:00.", "At 12 am.", "match"),
            ("О 12:00.", "At 12 p.m.", "match"),
            ("О 09:05.", "At 09:05.", "match"),
            ("О 09:05.", "At 09:05 AM.", "match"),
            ("О 21:05.", "At 9:05 a.m.", "mismatch"),
            ("О 21:05.", "At 9:05.", "review"),
            ("О 21:05.", "At 25:05.", "review"),
            ("О 21:05.", "At 123:05.", "review"),
            ("О 21:05.", "At 21:005.", "review"),
            ("О 21:05.", "At 21:05:00.", "review"),
            ("О 21:05.", "At 0:05 pm.", "review"),
            ("О 21:05.", "At 21:05 or 22:05.", "review"),
            ("О 21:05.", "At midnight.", "review"),
            ("О 21:05.", "At 21:05 or midnight.", "review"),
            ("О 21:05.", "At 21:05 UTC.", "review"),
            ("О 21:05.", "At 09:05 p.m.foo.", "review"),
            ("О 21:05.", "At x21:05x.", "review"),
            ("О 21:05 та 22:05.", "At 21:05.", "review"),
            ("О 9:05.", "At 09:05.", "review"),
            (None, "At 09:05.", "review"),
            ("О 19:00.", "At 7 p.m. in the evening.", "match"),
            ("О 19:00.", "At 7 p.m. in the morning.", "review"),
            ("О 19:00.", "At 7 p.m. Tomorrow morning we leave.", "match"),
            ("О 10:30.", "At 10:30.", "review"),
            ("О 12:30.", "At 12:30 (24-hour clock).", "match"),
            ("О 21:00.", "From 7–9 pm.", "review"),
        ]
        for source, translation, status in cases:
            with self.subTest(source=source, translation=translation):
                self.assertEqual(check(source, translation)["status"], status)

    def test_evidence_offsets(self):
        result = check("О 21:05.", "At 9:05 p.m.")
        for key, text in (("source", "О 21:05."), ("translation", "At 9:05 p.m.")):
            evidence = result[key]["evidence"][0]
            self.assertEqual(text[evidence["start"]:evidence["end"]], evidence["text"])
            self.assertEqual(result[key]["time"], "21:05")


class NormalizerTests(unittest.TestCase):
    def test_supported(self):
        cases = [('pl', 'o wpół do dziesiątej wieczorem', 'o 21:30'),
                 ('hr', 'u pola deset navečer', 'u 21:30'),
                 ('ru', 'в половине десятого вечера', 'в 21:30'),
                 ('uk', 'о пів на десяту вечора', 'о 21:30'),
                 ('uk', 'о пів на дванадцяту ранку', 'об 11:30'),
                 ('ru', 'в половине первого ночи', 'в 00:30')]
        for lang, phrase, expected in cases:
            with self.subTest(phrase=phrase):
                self.assertEqual(normalize_clock(lang, 'До: '+phrase+'. End.'), 'До: '+expected+'. End.')

    def test_unsupported_and_multiple(self):
        phrase = 'в половине десятого вечера'
        cases = [('xx', phrase), ('ru', 'обычный текст'),
                 ('ru', 'в половине неизвестного вечера'),
                 ('ru', 'буквав половине десятого вечера'),
                 ('ru', 'буквав половине десятого вечера и в половине второго'),
                 ('ru', phrase+' и '+phrase), ('ru', phrase+' и 22:30'),
                 ('ru', phrase+' и в половине второго'),
                 ('ru', 'в половине десятого вечерами'),
                 ('ru', 'в половине двенадцатого вечера')]
        for lang, text in cases:
            with self.subTest(text=text):
                self.assertEqual(normalize_clock(lang, text), text)
        with self.assertRaises(TypeError):
            normalize_clock('ru', None)

    def test_historical_checker_sha256(self):
        # Proves the published numeric checker is the actual experiment code.
        with open(clock_check.__file__, 'rb') as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        self.assertEqual(digest, '2296cd32cbc3cbca36b4ac0c9fd6022ab7cd2f767f7a8420249e45b96057e44f')

    def test_known_around_guard_limitation(self):
        # Existing checker accepts approximation; this is not a semantic guarantee.
        self.assertEqual(check('О 21:30.', 'Around 9:30 PM.')['status'], 'match')

    def test_cli_metadata_and_error(self):
        rows = [{'lang': 'ru', 'source': 'в половине десятого вечера', 'tag': 7}]
        result = subprocess.run([sys.executable, '-m', 'pipeline.clock_normalize', '-'],
                                input=json.dumps(rows), text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [dict(rows[0], source='в 21:30')])
        bad = subprocess.run([sys.executable, '-m', 'pipeline.clock_normalize', '-'],
                             input='{}', text=True, capture_output=True)
        self.assertNotEqual(bad.returncode, 0)

if __name__ == '__main__':
    unittest.main()
