# Pick and Place Launch Guide

Repository: https://github.com/elpis-lab/pick_and_place.git

branches: main, dev/uday
main for stable version
dev/uday - Currently in Dev 

This Guide is not generic yet.
Currently this is based on the running version in the ATHENA PC - RTX 4090. 

*** NOTE *** : NEVER CHANGE ANYTHING ON THE PC - UPDATE/UPGRADE WITHOUT INFORMING PROF/UDAY
FOR ANY ISSUE DONT TAKE YOUR OWN DECISION PLEASE CONTACT THE CURRENT MAINTAINER ON SLACK /EMAIL

Current Maintainer:
Uday Girish Maradana
umaradana@wpi.edu \
or on Slack  @ Uday

---
#### Launch the UR 10 Robot Driver


  	ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur10 robot_ip:=192.168.1.102 launch_rviz:=false


#### Launch the UR10 Moveit Description 

	ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur10 launch_rviz:=true use_mock_hardware:=true

----
#### Vision stack deployment

Python Orchestrator: Anaconda / Miniconda

Current deployment follows a individual Fast API based approach all communication using Async requests.

Base Folder: /home/athena/Projects/Pick_Place/Pick_Place_UR/

----

**Grasp Detection** - Any Grasp (Current Single Shot Grasp Detection - Not Tracking)
*Folder Path*: /home/athena/Projects/Pick_Place/Pick_Place_UR/anygrasp_sdk/grasp_detection
*Environment*: anygrasp_sdk

Commands:

Go to the folder path, Execute below commands

	conda activate anygrasp_sdk
  	python3 demo_api.py 


---

**LangSam** - For Segmentation using Language based prompt supports most of the Generic Objects
*Folder Path*: /home/athena/Projects/Pick_Place/Pick_Place_UR/lang_segment
*Environment*: lang_sam

Go the folder path, Execute below commands

	conda activate lang_sam 
  	python3 demo_api.py
    
---

**Gripper - RH P12 RNA** (python implementation is based on .so compiling of the existing C++ functionality)
*Folder Path*: /home/athena/Projects/Pick_Place/Pick_Place_UR/RH-P12-RN_Example/python
*Environemnt*: tossing_bot
For full class check the python rhp12 code for running 

Go the folder path, Execute below commands

	conda activate tossing_bot
  	python3 rh_p12_rna_api.py
    
    
before running this command ensure the USB is connected and the Driver - U2D2 (present on table) is switched on.

---

**For Place API**:
This takes the Eye to Hand Camera or Fixed Mounted Camera to detect where to place, this can also be used for picking objects.
*Folder Path*: /home/athena/Projects/Pick_Place/Pick_Place_UR
*Environment*: tossing_bot

Go the folder path, Execute below commands

	conda activate tossing_bot 
  	python3 place_api.py

---

#### MAIN CODE
For the **Pick Place DEMO**:
*Folder Path*: /home/athena/Projects/Pick_Place/Pick_Place_UR
*Environment*: Outside of Conda

*For Terminal based Version:*
	
  Run
  	
   	python3 main.py
      
*For Gradio based GUI:*

	python3 pick_place_gui.py
  

