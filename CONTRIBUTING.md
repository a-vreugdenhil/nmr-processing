# Working on this project

- Do work on a branch (`git checkout -b feature/short-name`), then open a pull request.
- Keep processing logic in `processing.py` (no GUI imports) so it can be tested.
- Add or update a test for any processing change.
- Run `ruff check .` and `pytest` before pushing.
- Tiny test datasets go in `tests/data/`; large Bruker datasets are not committed.
