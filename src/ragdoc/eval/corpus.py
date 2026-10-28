"""Deterministic synthetic technical-manual corpus used by tests, fixtures and the benchmark.

Every document has unique facts (prose value, table row, equation constants, code function) so ground truth is
unambiguous. This is a SYNTHETIC corpus: results on it do not transfer to real documents without re-measuring."""
from __future__ import annotations

import random
from pathlib import Path

TOPICS = ["regulator", "actuator", "sensor", "compressor", "valve", "inverter", "encoder", "relay", "pump", "controller"]
FILLER = [
    "Installation must follow the local electrical code and the instructions in this manual.",
    "Only trained personnel should service the enclosure after the supply has been isolated.",
    "Keep the ventilation openings clear of dust and make sure the mounting surface is flat.",
    "Inspect the connectors during every maintenance visit and replace any worn gaskets.",
    "The housing is made from powder coated aluminium and passes the standard vibration test.",
    "Record the serial number before contacting support so that the correct revision is identified.",
    "Use shielded cable for the signal lines and ground the shield at one end only.",
    "Store spare units in the original packaging in a dry location away from direct sunlight.",
    "Calibrate the device after any firmware update and document the result in the service log.",
    "Refer to the wiring diagram on the inside of the cover before connecting the terminals.",
    "The warranty is void if the seals are broken or the unit is opened by unauthorised persons.",
    "Cleaning should be done with a dry cloth and no solvents may be applied to the label.",
]


def _filler(rng: random.Random, n: int) -> str:
    return " ".join(rng.sample(FILLER, n))


def make_corpus(root: Path, seed: int = 7, n_docs: int = 24) -> list[dict]:
    rng = random.Random(seed)
    root = Path(root)
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "src").mkdir(parents=True, exist_ok=True)
    codes: set[str] = set()
    facts: list[dict] = []
    for i in range(n_docs):
        while True:
            code = f"{rng.choice('KRXZ')}{rng.choice('QWV')}-{400 + rng.randint(0, 999)}"
            if code not in codes:
                codes.add(code)
                break
        topic = TOPICS[i % len(TOPICS)]
        t_max = rng.randint(60, 140)
        f = {
            "i": i, "code": code, "topic": topic, "t_max": t_max, "t_store": t_max + rng.randint(20, 60),
            "volt": rng.choice([12, 24, 48]), "current": round(rng.uniform(0.5, 9.9), 1), "resp": rng.randint(2, 40),
            "gain": round(rng.uniform(1.1, 9.9), 1), "offset": round(rng.uniform(0.05, 0.95), 2),
            "md": f"docs/{topic}_{i}.md", "py": f"src/{topic}_{i}.py",
        }
        md = f"""# {topic.title()} {code} Technical Manual

## {code} Overview
{_filler(rng, 3)}

## {code} Thermal limits
The unit has a maximum operating temperature of {f['t_max']} degrees Celsius. {_filler(rng, 1)} The storage temperature may reach {f['t_store']} degrees Celsius.

## {code} Electrical specification
{_filler(rng, 1)}

| Parameter | Value | Unit |
| --- | --- | --- |
| Supply voltage | {f['volt']} | V |
| Rated current | {f['current']} | A |
| Response time | {f['resp']} | ms |

## {code} Control law
The output of the unit follows a linear transfer function.

$$ y = {f['gain']} \\cdot x + {f['offset']} $$

{_filler(rng, 2)}
"""
        py = f'''"""Firmware helpers for the {code} {topic}."""
import math

LIMIT_C = {f['t_max']}


def clamp_{topic}_{i}_temperature(value):
    """Clamp a reading to the rated limit of the {code} {topic}."""
    return min(value, {f['t_max']})


def {topic}_{i}_gain(x):
    """Apply the linear gain of the {code} {topic}."""
    return {f['gain']} * x + {f['offset']}
'''
        (root / f["md"]).write_text(md, encoding="utf-8")
        (root / f["py"]).write_text(py, encoding="utf-8")
        facts.append(f)
    return facts


def case_for(f: dict, ctype: str) -> dict:
    """A question with unambiguous ground truth for one fact."""
    code, topic = f["code"], f["topic"]
    if ctype == "prose":
        return {"ctype": "prose", "question": f"What is the maximum operating temperature of the {code} {topic}?",
                "expected_answer": f"maximum operating temperature of {f['t_max']} degrees Celsius", "expected_source": f["md"]}
    if ctype == "table":
        return {"ctype": "table", "question": f"What is the rated current of the {code} {topic}?",
                "expected_answer": f"Rated current | {f['current']} | A", "expected_source": f["md"]}
    if ctype == "equation":
        return {"ctype": "equation", "question": f"What is the transfer function of the {code} {topic}?",
                "expected_answer": f"y = {f['gain']} \\cdot x + {f['offset']}", "expected_source": f["md"]}
    if ctype == "code":
        return {"ctype": "code", "question": f"Which function clamps a temperature reading for the {code} {topic}?",
                "expected_answer": f"def clamp_{topic}_{f['i']}_temperature(value)", "expected_source": f["py"]}
    raise ValueError(ctype)
