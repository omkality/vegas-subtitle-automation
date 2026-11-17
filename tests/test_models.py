import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from vegas_subtitles.models import fetch


class ModelDownloadTests(unittest.TestCase):
    def registry(self, root):
        registry = root/"registry.json"
        registry.write_text(json.dumps({"tiny": {"repository": "org/model", "revision": "a"*40,
            "files": {"model.bin": {"bytes": 5, "sha256": hashlib.sha256(b"valid").hexdigest()}}}}))
        return registry

    def test_bad_remote_content_never_becomes_a_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("vegas_subtitles.models.urlopen", return_value=io.BytesIO(b"wrong")):
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    fetch("tiny", self.registry(root), root/"models")
            self.assertEqual(list((root/"models/tiny").iterdir()), [])

    def test_verified_download_is_reused_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = self.registry(root)
            with patch("vegas_subtitles.models.urlopen", return_value=io.BytesIO(b"valid")) as network:
                folder = fetch("tiny", registry, root/"models")
                self.assertIn("/resolve/"+"a"*40+"/", network.call_args.args[0])
            with patch("vegas_subtitles.models.urlopen", side_effect=AssertionError("Unexpected download")):
                fetch("tiny", registry, root/"models")
            self.assertEqual((folder/"model.bin").read_bytes(), b"valid")
