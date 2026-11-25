"""Keep the public Colab entry points and their displayed experiment results in sync."""

import contextlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
COLAB = "https://colab.research.google.com/github/Silveroboros-dev/slavic-english-slm/blob/main/"


class PublicNotebookTests(unittest.TestCase):
    def test_readme_links_to_every_public_notebook(self):
        readme = (ROOT / "README.md").read_text()
        for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
            with self.subTest(path=path.name):
                relative = path.relative_to(ROOT).as_posix()
                self.assertIn(f"]({relative})", readme)
                self.assertIn(f"]({COLAB}{relative})", readme)

    def test_all_defaults_reproduce_their_saved_stdout(self):
        # Walkthroughs 01–04 and experiment entry points 05–08 all run offline.
        # Exercise their checked-in defaults, including the seeded example in 02.
        paths = sorted((ROOT / "notebooks").glob("*.ipynb"))
        self.assertTrue(paths, "No public notebooks found")
        for path in paths:
            scope = {"__name__": "__main__"}
            with (
                patch("sys.path", list(sys.path)),
                patch.object(Path, "cwd", return_value=ROOT),
                patch("subprocess.run", side_effect=AssertionError("Unexpected subprocess")),
                patch("subprocess.Popen", side_effect=AssertionError("Unexpected subprocess")),
                patch("socket.create_connection", side_effect=AssertionError("Unexpected network")),
                patch("socket.socket.connect", side_effect=AssertionError("Unexpected network")),
            ):
                for index, cell in enumerate(json.loads(path.read_text())["cells"]):
                    if cell["cell_type"] != "code":
                        continue
                    with self.subTest(notebook=path.name, cell=cell.get("id", index)):
                        displayed = "".join(
                            "".join(output["text"])
                            for output in cell["outputs"]
                            if output["output_type"] == "stream"
                            and output["name"] == "stdout"
                        )
                        capture = io.StringIO()
                        with contextlib.redirect_stdout(capture):
                            exec(
                                compile("".join(cell["source"]), path.name, "exec"),
                                scope,
                            )
                        self.assertEqual(capture.getvalue(), displayed)
