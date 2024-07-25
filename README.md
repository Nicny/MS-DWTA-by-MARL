**Introduction:**
====

This code utilizes multi-agent reinforcement learning methods to solve the multi-ship dynamic weapon-target assignment problem. It is noteworthy that our work is simulated on the Mozi simulation platform rather than merely numerical calculations. Therefore, it is necessary for you to be familiar with the usage and relevant interfaces of this platform. The code offers a variety of algorithms to choose from, and its framework is based on PYMARL, making it easy to integrate other multi-agent reinforcement learning algorithms.

**Preparations:**
====

**Mozi Platform:** Since the code needs to integrate with the Mozi simulation platform, you first need to download the platform (website: https://www.hs-defense.com/col.jsp?id=105). After downloading, you need to successfully install it on your computer, preferably on the D drive. Ensure that your Mozi platform is functioning properly. You can learn how to use the Mozi platform, such as unit placement and scenario creation, on its official website.

**Mozi Python Interface:** You also need to download the interface files from the official website, which allow you to control the platform using Python code. These files are essential, as you cannot perform any operations through Python without them. We recommend you first familiarize yourself with some basic functions of controlling Mozi with Python, such as planning aircraft movements and controlling ship firing. Due to the high degree of openness of these functions, most issues encountered while using the code may stem from interface errors.

**Some Useful Modifications:** We provide some helpful tips and modifications to facilitate the use of the platform.

* Ensure that the paths involved in the code are consistent with the actual installation path of the platform.

* The platform needs to be switched to AI training mode, which can be found and opened in ".\Mozi\MoziServer\bin\ConfigSet.exe".

* Since some information cannot be directly obtained through the interface, we currently consider using stored simulation logs to acquire it. Therefore, you need to adjust the message output limit to 1000 in the Mozi platform and select only combat unit AI, weapon terminal calculations, weapon logic, and weapon firing. Excessive message output would make reading inconvenient. You can select other message outputs as needed.

* The platform records not only simulation information but also other data, which may occupy a lot of disk space. We suggest modifying the configuration file in the Mozi to disable unnecessary storage options.

*  Our experimental scenarios are limited to defense, using only missile defense against opponent missiles. To avoid other situations, you need to remove redundant weapons and sensors from both red and blue sides that may interfere with the scenario. The red side needs to deploy an offensive mission against the blue side, while the blue side, under our control, needs to remain stationary and in manual fire mode.

* Parameter Modification: Set the red side's aircraft maneuverability to 500 and adjust the blue side's weapon range as needed to avoid erroneous defense scenarios. For specific operations, refer to the Mozi platform database modifications.

**Code Architecture:**
====

Our code is based on the PYMARL framework, including:

`arguments.py`: Records environment and algorithm parameters.

`agent.py`: Agent creation and action selection.

`env.py`: Environment.

`replay_buffer.py`: Experience pool for storage and sampling.

`rollout.py`: Runs one episode, including observation, action selection and execution, reward calculation, and data storage for each step.

`runner.py`: Runs multiple episodes, trains, and periodically evaluates.

`network folder`: Stores files for establishing basic networks for different algorithms.

`policy folder`: Stores files for executing training of different algorithms.

`result folder`: Stores experimental data, images, and neural network models.

**Running the Code:**
====

Run the `main.py`.

**Citation:**
====

Our work is not targeted at any organization or country, and there is no conflict of interest.

If you find our work helpful, please cite our paper.

**Paper Name:** `Multi-Ship Dynamic Weapon-Target Assignment via Cooperative Distributional Reinforcement Learning with Dynamic Reward.`

If you have any questions, please leave a comment in the issue section. Thank you!

