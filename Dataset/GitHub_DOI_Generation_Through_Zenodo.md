# GitHub DOI Generation Through Zenodo

## Recommended Method: GitHub + Zenodo

This is the most commonly used method for generating a DOI for research code hosted on GitHub.

## Step 1: Prepare Your GitHub Repository

Ensure that your GitHub repository is **public**.

Add the following files and content:

- `README.md` — include the project description, usage instructions, and requirements.
- `LICENSE` — add a suitable license, such as MIT or Apache 2.0.
- Commit all code corresponding to the version used in the research paper.

## Step 2: Create a Release on GitHub

1. Open your GitHub repository.
2. Go to **Releases**.
3. Click **Draft a new release**.
4. Assign:
   - A version number, such as `v1.0.0`.
   - A brief description of the release.
5. Click **Publish release**.

> [!IMPORTANT]
> This step is critical because Zenodo assigns DOIs to releases, not branches.

## Step 3: Link GitHub to Zenodo

1. Sign in to [Zenodo](https://zenodo.org) using your GitHub account.
2. Go to **Account → GitHub**.
3. Enable Zenodo access for your repository.
4. Toggle the repository **ON**.

## Step 4: Mint a DOI

Create or update a GitHub release.

Zenodo will automatically:

- Archive the release.
- Generate a DOI.

Zenodo provides:

- A **version-specific DOI**, which is recommended for research papers.
- A **concept DOI**, which points to all versions of the software.

## Step 5: Add Citation Metadata

Zenodo automatically generates a citation, but you should verify the following information:

- Authors
- Title
- Year
- Version
- DOI

Optionally, add a `CITATION.cff` file to your GitHub repository so GitHub displays a **Cite this repository** button.

## How to Cite GitHub Code in a Research Paper

### In-Text Citation Example

> The machine learning models were implemented using custom Python code [1].

### Reference List Example

Use the following format, based on the Scientific Reports style:

```text
1. Author(s). Title of software. Version. Zenodo. https://doi.org/xx.xxxx/zenodo.xxxxx
```

> [!NOTE]
> Always cite the DOI, not the raw GitHub URL.
