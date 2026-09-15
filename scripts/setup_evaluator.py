"""Prepare and optionally start the local evaluator demonstration.

The script uses only the Python standard library until the project virtual
environment and the pinned application dependencies have been installed.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import venv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepara a demonstração local completa do Streamlit."
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Inicia o Streamlit em local_preview após a validação.",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Reutiliza o ambiente virtual sem reinstalar as dependências.",
    )
    return parser


def _venv_python() -> Path:
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def _run(label: str, command: list[str], *, env: dict[str, str] | None = None) -> None:
    print(f"\n==> {label}", flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if sys.version_info[:2] != (3, 12):
        raise SystemExit(
            "Python 3.12 é obrigatório. Use 'py -3.12' no Windows ou "
            "'python3.12' no Linux/macOS."
        )

    python = _venv_python()
    if not python.is_file():
        print("\n==> Criando o ambiente virtual com Python 3.12", flush=True)
        venv.EnvBuilder(with_pip=True).create(VENV)

    if not args.skip_install:
        _run("Atualizando o pip", [str(python), "-m", "pip", "install", "--upgrade", "pip"])
        _run(
            "Instalando a aplicação local",
            [
                str(python),
                "-m",
                "pip",
                "install",
                "-e",
                ".",
                "-r",
                "requirements/app-local.txt",
            ],
        )

    _run(
        "Baixando datasets públicos (downloads grandes podem ficar alguns minutos sem saída)",
        [str(python), "scripts/acquire_data.py", "--dataset", "all"],
    )
    _run("Processando e anonimizando os dados", [str(python), "scripts/preprocess_data.py"])
    _run("Construindo o banco de pacientes somente leitura", [str(python), "scripts/build_patient_database.py"])
    _run("Construindo o índice RAG local", [str(python), "scripts/build_rag_index.py"])
    _run("Validando a aplicação Streamlit", [str(python), "scripts/validate_streamlit_app.py"])

    print(
        "\nDemonstração local pronta. Resultado esperado: ok=true.\n"
        "Para iniciar depois, use:\n"
        f"  {python} -m streamlit run app/streamlit_app.py",
        flush=True,
    )
    if args.run:
        local_env = os.environ.copy()
        local_env["TECHCARE_EXECUTION_MODE"] = "local_preview"
        local_env.pop("TECHCARE_REMOTE_URL", None)
        local_env.pop("TECHCARE_REMOTE_TOKEN", None)
        _run(
            "Iniciando o Streamlit em http://localhost:8501",
            [str(python), "-m", "streamlit", "run", "app/streamlit_app.py"],
            env=local_env,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
