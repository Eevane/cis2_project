# cis2_project
Project: Multilateral Teleoperation based Haptic Training System for the dVRK for EN 601.456 Computer Integrated Surgery II

## How to Apply Our Code
This repo contains modified versions of several files from the original [jhu-dVRK](https://github.com/jhu-dvrk) repositories.

First is to install and comilation of ROS2 and related repositories, the instruction can be found on (https://dvrk.readthedocs.io/2.3.0/pages/software/compilation/ros2.html), using the development branches

After the compilation, simply copy the files in `scripts/` to their corresponding paths:

```bash
cd /path/to/cis2_project
cp scripts/jhu-dVRK/* ~/ros2_ws/src/dvrk/jhu-config-dvrk/jhu-dVRK/
cp scripts/jhu-daVinci/* ~/ros2_ws/src/dvrk/jhu-config-dvrk/jhu-daVinci/
```

## How to run the teleoperation
Our code provide two ways of teleoperation: single console and 2 consoles

For single console, which is using 2MTMs controlling 1 PSM:
```bash
source ~/ros2_ws/install/setup.bash
cd ~/ros2_ws/src/dvrk/jhu-config-dvrk/jhu-dVRK/
ipython3
```
```python
run multi-teleoperation-sin.py
# asking for input json file
console-MTML-PSM1-PSM2-Teleop.json
```
After the alignment is compoleted, then press the Coag pedal and the teleoperation will start.

For the multilateral teleoperation on 2 consoles (one through firewire and the other through Ethernet), which is using 4MTMs controlling 2 PSMs:
```bash
source ~/ros2_ws/install/setup.bash
cd ~/ros2_ws/src/dvrk/jhu-config-dvrk/jhu-dVRK/
ipython3
```
```python
run multi-teleoperation.py
# asking for first input json file
console-MTML-MTMR-PSM1-PSM2-Teleop.json
# asking for second input json file
../jhu-daVinci/console-MTML2-MTMR2.json
```
After the alignment is compoleted, then press the Coag pedal and the teleoperation will start.
