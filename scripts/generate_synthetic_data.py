"""Generate stage 3 internal data for the fictional Hospital TechCare."""

from __future__ import annotations

import json

from clinical_assistant.config import project_root
from clinical_assistant.synthetic.generator import generate_synthetic_hospital


if __name__ == "__main__":
    result = generate_synthetic_hospital(project_root())
    print(json.dumps(result, ensure_ascii=False, indent=2))
