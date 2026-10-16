# Attribution and upstream licensing status

The model and experimental design are credited to **Jeongho Park and Kyuri Jo**,
authors of *DriverMONI: Cancer Driver Gene Prediction With Multimodal Deep Learning
Integrating Multiomics Data and Condition-Specific Network Information*.
[Paper DOI](https://doi.org/10.1109/TCBBIO.2025.3643887).

Reference implementation:
[BML-cbnu/DriverMONI](https://github.com/BML-cbnu/DriverMONI), commit
`ddb0a830158ba41c90019bd5ca92054c2ba0869b`.

On 2026-09-27, inspection of the pinned Git tree, README, source headers, and
reference-folder readmes found no LICENSE/COPYING file or explicit redistribution
grant. The public GitHub landing page also did not identify a license. This is
an absence-of-permission finding, not a claim that the authors forbid all use.
Public availability should not be treated as a permissive license; see
[GitHub's licensing guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository).

Accordingly, this Git project contains independently written implementation and
tests, attribution, source locations, and hashes. Previously bundled verbatim
upstream model/dataset test excerpts were removed before the first commit.
Optional tests read selected classes from a user-supplied local checkout in memory,
after checking the inspected source hashes. No generated copies are saved or
included in the repository. The upstream files themselves are not modified.
This is a source-informed reimplementation, not an independently staffed
clean-room implementation or a legal certification of redistribution rights.

Original processed BRCA data, HumanNet files, neighbor caches, original source,
raw TCGA data, and the paper PDF are not distributed with this project. Teammates obtain
the necessary files directly from the authors using [data/README.md](data/README.md).
That acquisition does not confer downstream redistribution rights. Upstream and
underlying data-provider terms remain applicable; obtain permission before
repackaging those materials for others.

Python dependencies are installed separately through pip and retain their own
licenses. This notice does not relicense upstream code, data, or dependencies.
No open-source license for the group's own contributions has been selected in
this task; the group should decide that before presenting a public release as
open source. A citation by itself does not replace a license or permission.
