"""Turn scripts/copy_dk_lines.js into a one-line javascript: URL (prints it, and writes scripts/bookmarklet.txt)."""
import re, pathlib
src = pathlib.Path(__file__).with_name("copy_dk_lines.js").read_text()
src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)                  # strip the comment block
one = " ".join(line.strip() for line in src.splitlines() if line.strip())
url = "javascript:" + one
pathlib.Path(__file__).with_name("bookmarklet.txt").write_text(url + "\n")
print(url)
