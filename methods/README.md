# methods/

One folder per method. Numbered folders follow the shortlist in `../paper/research_survey.md`, from easiest to most ambitious.
Copy `_template/README.md` when adding a method.

Convention for each method:
- Upstream code is cloned into `<method>/upstream/` (gitignored if large; record the commit in the README).
- Our own wrappers and configs live next to it.
- Raw outputs go to `../runs/` (gitignored). Summaries go to `../results/` and `../paper/work_log.md`.
