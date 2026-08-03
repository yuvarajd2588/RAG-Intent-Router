from pathlib import Path
import csv
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "phase1-data-normalize"))

from normalize import normalize_text


def test_normalize_text_expands_common_chat_shorthand():
    assert normalize_text("pls help me file a claim") == "please help me file a claim"
    assert normalize_text("can u help me") == "can you help me"
    assert normalize_text("ASAP claim update") == "as soon as possible claim update"


def test_normalize_text_handles_punctuation_and_whitespace():
    assert normalize_text("  How   do I file?  ") == "how do i file"


def test_normalize_text_keeps_department_label_logic_outside():
    assert normalize_text("claim status pls") == "claim status please"
