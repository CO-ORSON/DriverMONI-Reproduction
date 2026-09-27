# Fresh-teammate review — 2026-09-27

Reviewed the publishable repository files, README commands, dependency installation,
data acquisition/schema instructions, entry points, configuration, device handling,
seed handling, and tests. Scientific model modules, graph semantics, optimizer,
loss, features, labels, and default experiment settings were not changed.

## Findings and fixes

| Finding | Fix |
| --- | --- |
| `python scripts/debug_brca.py` failed to import `src`; only module invocation worked. | Added project-root import setup, consistent with the other scripts. Direct CLI paths work from an unrelated directory. |
| Library/debug imports without DGL configuration attempted to write `~/.dgl/config.json`; this failed in the restricted environment. A pre-existing non-PyTorch backend could also break execution. | Select PyTorch in package initialization before importing DGL, without writing user configuration. An explicitly incompatible backend gives an actionable error. |
| Config accepted seeds up to `2**32-1`, but DGL raises a C integer overflow for seeds >=`2**31`. | Validate the common supported interval `[0,2**31)` before seeding. Default 42 is unchanged; the maximum accepted value is tested. |
| Misspelled nested config keys could be ignored; missing/type-invalid values produced confusing low-level errors. | Validate required/unknown keys, numeric values, path strings, device syntax, and fixed upstream metadata before data loading. |
| Missing data failed on the first CSV without explaining the other required files. | Report all missing roles and resolved paths together, accept the documented ZIP alternative, and link acquisition instructions. Disabled filter/cache options do not require unused files. |
| Quoted `~` paths were not consistently expanded. | Expand home-relative CLI config/data/output paths and per-file input paths consistently. |
| YAML root/file paths and CLI paths had different bases but insufficient explanation. | Document every path base in README and `configuration.md`; input overrides and space-containing paths were exercised through the full runner. |
| Model/graph YAML looked like tunable architecture definitions despite fixed implementation. Several config constraints and consequences were scattered. | Add a complete configuration reference identifying fixed assertions, configurable values, defaults, scientific departures, environment variables, and smoke-test scope. |
| Setup did not state the macOS wheel floor or distinguish direct pins from a full lock. | Document macOS 14+ for the selected SciPy arm64 wheel, Python 3.12, CPU validation scope, CUDA wheel requirements, and the optional original full dependency lock. |
| Source-hash checks could fail after automatic CRLF conversion. | Acquisition commands use `git -c core.autocrlf=false clone` for the separately obtained original source. |

No missing dependency was found after a genuinely fresh installation. Pydantic is
retained and explained because DGL GraphBolt imports it even though project code
does not. No user-specific absolute paths were found in publishable source/config;
paths derived from `__file__` are portable. Absolute paths in generated provenance
are expected and those reports remain ignored by Git.

## Verification

- Created an empty Python 3.12 virtual environment outside the project and
  installed `requirements.txt` from package indexes, without reusing the project's
  existing environment. Installation succeeded; `pip check` found no broken
  requirements. One transitive package, filelock, resolved to 4.0.4 rather than
  the original lock's 4.0.3, illustrating why the full lock is separately provided.
- In that new environment, the full tests with pinned external original source
  and BRCA data returned **78 passed, 1 skipped in 23.59 seconds**. The only skip
  is CUDA, unavailable on this host.
- Without external source/data: **55 passed, 24 explicitly skipped**. Original
  classes remain external and are loaded only for optional parity checks.
- Copied only Git-shareable files into a separate directory containing a space,
  then placed only the six documented data/reference files under `data/upstream`.
  This was a relocation simulation, not a clone of a published group remote.
- In the relocated copy with the fresh environment and no DGL backend variable,
  direct debug execution produced BRCA1's expected 491 nodes, 3,398 edges, 12 DNN
  features, 13 node features, and driver label.
- The relocated **real-data** smoke command completed three folds and two epochs
  on 12 targets. Its fold assignments, per-epoch losses, selected epochs,
  probabilities, and metrics exactly matched the earlier seed-42 smoke run.
- The full `run_brca.py` entry point also completed a three-fold, two-epoch
  **synthetic-data** experiment from outside the project directory, with custom
  filenames, a separate YAML path, CLI root/output overrides, and paths containing
  spaces. Disabled comparison/cache settings did not require those files.
- New regression tests cover entry-point import setup, backend configuration,
  seed bounds, invalid config values/keys, and aggregated missing-data errors.
  CLI help tests execute via `runpy`; actual fresh-process commands were verified
  separately. Nested Python subprocess imports hit this sandbox's OpenMP shared
  memory restriction, while directly launched processes succeeded; no OpenMP
  bypass or scientific runtime setting was added to production code.

The prior complete 30-epoch BRCA results were preserved. A full experiment was not
rerun for these portability-only changes; controlled parity, fresh installation,
real-input smoke, and full-CLI checks cover the changed paths.

## Remaining limits

- CPU/macOS arm64 is verified; Windows, Linux and CUDA installations still need
  platform-specific validation. The documented macOS lock is not a CUDA lock.
- Source mode behavior and test-fold epoch selection remain intentional; published
  scores, historical seeds/splits and an exact original training trajectory are
  not established by these checks.
- This local project has no published remote or initial commit. The README group
  clone URL is explicitly a placeholder to replace when the group publishes it.
- Original source/data redistribution permission remains unresolved. Those files
  are ignored and obtained separately; group code licensing remains a group decision.
