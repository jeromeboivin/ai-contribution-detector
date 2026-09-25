from unidiff import PatchSet

from aicontrib.diff.commit import _post_image_text

SAMPLE_DIFF = """\
diff --git a/foo.py b/foo.py
index e69de29..4b825dc 100644
--- a/foo.py
+++ b/foo.py
@@ -1,2 +1,3 @@
 def foo():
-    pass
+    return 42
+    # trailing comment
"""


def test_post_image_reconstruction():
    patch = PatchSet(SAMPLE_DIFF)
    patched_file = patch[0]
    text, lines_changed = _post_image_text(patched_file)
    assert "def foo():" in text
    assert "return 42" in text
    assert "pass" not in text
    assert lines_changed == 3  # 1 removed + 2 added
