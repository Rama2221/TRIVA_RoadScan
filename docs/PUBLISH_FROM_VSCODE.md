# Publish the prepared folder from VS Code

Open **RoadScan_Public_Repo**, not the parent delivery directory or its sibling experiment archive. This folder has a local Git repository and local Git LFS filters configured; no remote, commit or push was created by the packaging step.

1. Open VS Code Source Control and review the files. `.local_artifacts`, runtime state, environments and caches must remain ignored.
2. Commit the reviewed files, then choose **Publish to GitHub** and **Public**. If prompted for a repository name, use `RoadScan` or another available name.
3. The RF checkpoint is larger than a normal GitHub file upload permits, so keep `.gitattributes` and Git LFS enabled. Publish through Git, not by dragging the model file into the website uploader.
4. Publish the exact curated data as a Release asset. With GitHub CLI authenticated and the terminal in this repository:

```powershell
gh release create v1.0-data .local_artifacts/RoadScan_Dataset.zip --title "RoadScan curated dataset" --notes "Exact 8,000 RDD + 100 Saudi dataset. Sources and limitations: data/metadata/."
```

If the Release already exists, upload the asset using the GitHub Release page or `gh release upload v1.0-data .local_artifacts/RoadScan_Dataset.zip`. Do not substitute a differently curated dataset under this filename. Check the recorded SHA-256 in `data/dataset_release.json`.

5. Verify from a fresh clone: `git lfs pull`, install dependencies, run artifact verification, import/download data, and start the app as described in the README.

The public URL does not exist until you publish. `scripts/download_dataset.py` derives the Release URL from the repository's actual GitHub remote, or accepts `--url` explicitly. It does not invent a hosted link.

The sibling previous-experiment folder is optional supporting material. To publish it separately, initialize its own repository and enable Git LFS before committing its ZIP and checkpoint files. Its original notebook outputs have been retained.
