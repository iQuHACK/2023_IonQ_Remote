# 2023_planning_ionq
IonQ's repository for iQuHACK 2023 (private planning)


## Working on qBraid
[<img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" width="150">](https://account.qbraid.com?gitHubUrl=https://github.com/varshaneya/2023_IonQ_Remote/tree/Karan)
1. If you're working on qBraid, first fork this repository and click the above `Launch on qBraid` button. It will take you to your qBraid Lab with the repository cloned.
2. Once cloned, open terminal (first icon in the **Other** column in Launcher) and `cd` into this repo. Set the repo's remote origin using the git clone url you copied in Step 1, and then create a new branch for your team:
```bash
cd  <ionq_git_repo_name>
git remote set-url origin <url>
git branch <team_name>
git checkout <team_name>
```

3. Use the environment manager (**ENVS** tab in the right sidebar) to [install environment](https://qbraid-qbraid.readthedocs-hosted.com/en/latest/lab/environments.html#install-environment) "IonQ". The installation should take ~2 min.
4. Once the installation is complete, click **Activate** to [add a new ipykernel](https://qbraid-qbraid.readthedocs-hosted.com/en/latest/lab/kernels.html#add-remove-kernels) for "IonQ".
5. From the **FILES** tab in the left sidebar, double-click on the IonQ repository directory.
6. Open and follow instructions in [`verify_setup.ipynb`](verify_setup.ipynb) to verify your "IonQ" environment setup.
7. You are now ready to begin hacking! Work with your team to complete the challenges.

For other questions or additional help using qBraid, see [Lab User Guide](https://qbraid-qbraid.readthedocs-hosted.com/en/latest/lab/overview.html), or reach out on [Discord](https://discord.gg/gwBebaBZZX).
