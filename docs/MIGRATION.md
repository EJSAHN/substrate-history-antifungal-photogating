# Migration from the initial submission code

Use `scripts/run_analysis.py`; the old runner and overlapping HSI wrappers are
retired from the active tree. The old code remains in repository history.

- Added the verified R1 factorial, matched-control, exact spectral and sensitivity analyses.
- Removed approximate variable clustering from active computation and exports.
- Replaced arbitrary CSV-to-worksheet globbing with an explicit workbook schema.
- Removed dependence on local recovery paths, original drive letters and source-object numbering.
- Kept raw data, historical outputs and manuscript material out of this repository.
- Separated figure and Word-document generation into the authors' local package.
- Completed the standard MIT text already designated by the old LICENSE, whose
  permission paragraph was truncated. The existing copyright line is unchanged.

The public and local packages share the same `src/uvsm` numerical files. An upload
package is not evidence that the remote repository was updated. Apply a reviewed
commit; do not delete the repository or rewrite earlier history.
