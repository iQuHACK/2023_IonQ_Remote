# 2023 iQuHack IonQ Remote Challenge

# Team "TheUntitledTeam"

# Members 

* agercas
* carleseb
* Gopal-Dahale
* ichirokira
* Loe-web

# Write-up
For the first part, we tried using FRQI and NEQR (as described [here](https://qiskit.org/textbook/ch-applications/image-processing-frqi-neqr.html)) for image encoding. 
At the moment, these are considered state-of-the-art quantum image encoding methods. However, we quickly noticed that although the encoding was producing good results, it was extremely inefficient (encoding a single image took 2-4 minutes). 
Hence, we decided to try Amplitude encoding instead (as described [here](https://qiskit.org/textbook/ch-applications/quantum-edge-detection.html)). 
Early on we made a decision to rescale the images from 28x28 to 16x16 and use 9 qubits for encoding. These produced very good results, so we carried on. 
In hindsight, this was a small mistake. Since we can use up to 16 qubits, it would have been better to either resize the image to 32x32 and use all 16 qubits for encoding. 
Or even better, we could have padded the image with zeros to create an embedded 32x32 image and use all 16 qubits (without any loss of information). 
By the time we realized this, we had already trained the classifier, and we didn't have time to change it.

The other problem with our solution for part 1 is that we are using 9 qubit gate (instead of multiple 2-qubit gates). Circuit decomposition was taking too long for this (~4 mins) and we didn't have enough time to implement this from the 2-qubit gates. 
Because of this reason we disabled the assertion in the quantum gate calculation method. 

TODO: 
second part

## Working on qBraid
[<img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" width="150">](https://account.qbraid.com?gitHubUrl=https://github.com/iQuHACK/2023_planning_ionq.git)
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
