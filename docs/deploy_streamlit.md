# Put the results viewer online

This deployment shares the group's recorded results, prepared gene measurements
and model explanation through Streamlit Community Cloud. The online viewer has
four pages: **Experiments**, **Gene explorer**, **Model architecture** and
**About this app**. It does not train a model or calculate new predictions.
The full app, including B/D live prediction, remains available
[on your own computer](streamlit_app.md).

This guide describes the prepared deployment workflow. An actual Community
Cloud deployment has not yet been verified; a successful local run alone does
not establish that the hosted app works.

Preparation verified on 26 October 2026: **300 project tests passed, 2 skipped**;
the **6 standalone viewer tests also passed in a fresh Python 3.12 environment**
containing only the cloud dependencies and pytest. All four viewer pages ran
without importing PyTorch or DGL. The dependency resolver found compatible
Linux x86_64 / Python 3.12 wheels for the complete cloud dependency set. This is
an installation-compatibility check, not a completed Linux hosting test.

## What will be available online

| Feature | Online viewer |
| --- | --- |
| Original, A, B, C and D experiment results | Available from the five published JSON snapshots |
| D versus its matched fixed C control | Available, with interpretation and limitations |
| Gene measurements for the updated dataset | Available from `data/brca/BRCA_input_data.csv` |
| Model architecture | Available |
| Original dataset's individual gene measurements | Unavailable; the authors' original CSV is not included in this upload |
| Saved scores for individual genes | Unavailable; full per-gene result reports stay local |
| New live predictions | Unavailable; model checkpoints and the full inference inputs stay local |

The gene explorer lets a viewer inspect the existing 13 measurements for a gene.
It does not turn those measurements into a new driver score. Aggregate results
describe the recorded experiments, with the original/A test-selection and D
matched-control caveats kept visible. See the [results index](../results/README.md)
for the scientific interpretation.

## 1. Put the app files on the deployment branch

Merge the Streamlit pull request into the GitHub repository, then open
[CO-ORSON/DriverMONI-Reproduction](https://github.com/CO-ORSON/DriverMONI-Reproduction)
and select `main`. Confirm that these files are present:

```text
cloud/app.py
cloud/requirements.txt
streamlit_app.py
.streamlit/config.toml
src/app_data.py
src/app_results.py
src/feature_schema.py
data/brca/BRCA_input_data.csv
results/published/original.json
results/published/a.json
results/published/b.json
results/published/c.json
results/published/d.json
```

The repository also includes supporting files and the local-app code. There is
no need to upload your virtual environment, model checkpoints, raw TCGA data,
full experiment reports or credentials.

For a preview before merging, deploy the `streamlit-app` branch instead. Use
`main` for the shared version after merging, so future main-branch updates reach
the hosted app.

## 2. Connect Streamlit to GitHub

1. Visit [Streamlit Community Cloud](https://share.streamlit.io).
2. Sign in, or create an account. You can use GitHub sign-in.
3. Connect the GitHub account that owns the repository: `CO-ORSON`.
4. If prompted, authorize Streamlit to access your public repositories.
5. Select the `CO-ORSON` workspace.

Signing in through GitHub and granting repository access are separate steps.
Streamlit requires repository admin access to deploy an app; the repository
owner has that access. See the official
[GitHub connection instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started/connect-your-github-account).

## 3. Create the app

Click **Create app**, then **Yup, I have an app** if asked. Fill in:

| Field | Value |
| --- | --- |
| Repository | `CO-ORSON/DriverMONI-Reproduction` |
| Branch | `main` after merging; `streamlit-app` only for a preview |
| Main file path | `cloud/app.py` |
| App URL | Optional: choose an available name, such as `drivermoni-brca` |

If a value is missing from a dropdown, enter it manually. Select **Advanced
settings**, choose **Python 3.12**, and click **Save**. This viewer needs no
secrets, API keys or passwords; leave the Secrets field empty. Then click
**Deploy**. These controls follow Streamlit's
[deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

Use `cloud/app.py` as the entrypoint. It opens the shared viewer and uses
`cloud/requirements.txt`, which contains only the viewing dependencies.
The root `streamlit_app.py` entrypoint opens the full local interface and is
not the entrypoint for this deployment.

## 4. Check the hosted app

Wait for the dependency installation and startup to finish. When the app opens:

1. Confirm the sidebar shows the four viewer pages, without **Live prediction**
   or local model-setup controls.
2. On **Experiments**, switch through Original → A → B → C → D. D should show
   mean AUC around **0.921** and F1 around **0.207**, with its separate matched
   control. These are saved results, not measurements from a new cloud run.
3. Open **Gene explorer**, choose an updated-data experiment and find `BRCA1`
   or `TP53`. Open the input-measurement table and confirm it displays 13 rows.
4. Open **Model architecture** and check that its diagram appears.
5. Open **About this app** and confirm that the viewer's scope is clear.

If the app fails to start, select **Manage app** to read its logs. The error
message usually identifies a missing file or dependency. Streamlit documents
the log controls in its
[app-management guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app).

## 5. Share the link

After those checks pass, use **Share → Copy link** and send that URL to your
teammates. A deployment from this public repository is public by default. To
limit viewing, open the app's **Settings → Sharing** and choose access for
specific people. Streamlit currently permits one private app per account; see
its [sharing documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app)
for the current rules and viewer sign-in process.

The hosted app has a `streamlit.app` address. The local address
`http://localhost:8501` works only on the computer running the local app and is
not the link to share with the group.

## Why the deployment has its own requirements file

Community Cloud checks the entrypoint's directory for a supported dependency
file before checking the repository root. With entrypoint `cloud/app.py`, it
finds `cloud/requirements.txt` and installs the viewing dependencies. This avoids
installing the full PyTorch/DGL training environment for a results viewer.
`requirements-app.txt` is for local installation and is not an automatically
discovered cloud dependency filename. See Streamlit's
[dependency discovery rules](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).

Community Cloud runs the app from the repository root. The shared theme stays
at `.streamlit/config.toml`, and the launcher resolves the project files from
that checkout. The shared configuration does not force a local-only server
address. See the official
[file-organization guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization).

You can preview the same viewer locally, from the project root, using a
Python 3.12 environment:

```bash
python -m pip install -r cloud/requirements.txt
python -m streamlit run cloud/app.py --server.address 127.0.0.1
```

## Later updates and common problems

Changes pushed to the deployed branch are picked up by Community Cloud;
dependency changes trigger installation again. Updating a tracked result file
changes the displayed saved experiment, not the model itself. Review result
exports before publishing them. Streamlit describes this behavior in its
[app-management guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app).

| Problem | Check |
| --- | --- |
| Repository is not listed | Connect the `CO-ORSON` GitHub account, select that workspace, or enter the repository manually. |
| It tries to install PyTorch or DGL | Confirm the main file is `cloud/app.py` and `cloud/requirements.txt` exists on the selected branch. |
| Live prediction appears online | The deployment is using the local entrypoint; use `cloud/app.py`. |
| Published results cannot load | Confirm all five `results/published/*.json` files are present and unchanged from the reviewed export. |
| Updated gene measurements are missing | Confirm `data/brca/BRCA_input_data.csv` is committed on the deployed branch. |
| Original measurements or individual gene scores are unavailable | This is the viewer's intended scope; use the local app and its documented extra files for those features. |
| Python was selected incorrectly | Delete and redeploy with Python 3.12; changing Python requires redeployment. |

The Python-version procedure is documented in Streamlit's
[upgrade guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/upgrade-python).
