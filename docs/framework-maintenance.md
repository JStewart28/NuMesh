# Maintaining the Tessera framework

Read this when adding a system, changing the build/run plumbing, or planning a
large change. The never-violate invariants stay in [CLAUDE.md](../CLAUDE.md).

---

## Adding a system

1. Create `systems/<system>/claude.md` with **all seven required sections**
   (Environment, Build-config args, Build command, Run command, Batch template,
   Non-test binaries, Backends).
2. Add a row to the hostname table in [CLAUDE.md](../CLAUDE.md).
3. Add a `case` branch in `scripts/lib/tessera_env.sh`.
4. Add `scripts/<system>/profile.defaults.sh` (committed).
5. Add `scripts/<system>/runtime_env.sh` if launch-time env vars are needed.
6. Add `scripts/<system>/run_regression_minset.<scheduler>`.
7. Declare the system's backends in both
   [build-and-run.md](build-and-run.md#compute-backends) and
   `systems/<system>/claude.md`.
8. Commit an env snapshot under `systems/<system>/` if one is used (e.g. spack.yaml).

---

## Planning large changes

**Checkpoint commits in plans.** Break large changes into stable, reviewable
commits at intermediate milestones.
