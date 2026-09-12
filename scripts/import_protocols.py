"""Import local protocols from the controlled inbox."""

from clinical_assistant.acquisition.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["--dataset", "protocols"]))

