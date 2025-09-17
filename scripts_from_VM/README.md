Google Cloud Notes:
 - **to start/stop the VM:**
   gcloud compute instances [stop/start] yolo-train --zone=us-central1-a
- **After starting the the VM, make sure you run this command to set up the ssh(needs to be run everytime the VM is stopped and started again):**
  gcloud compute config-ssh
- **to list all the current instances**
  gcloud compute instances list
- **SSH to the VM:**
  ssh yolo-train.us-central1-a.seaqr-detection-123


**Structure of the current data:**
All data could be found here on the VM: /data/
- dataset/
  - images/
    - train/
    - test/
    - validation/
  - labes/
    - train/
    - test/
    - validation
  -classes.txt
  - data.yaml
  - yolo11n.wt
 
  #NOTE: as of 9/16/2025 the data of KOLOMVERSE + whales are combined and in YOLO11n format. In addition, moost of the images are labeled except around 36k images that are just the background images(negative images).

All the python packages that are needed to run the training on the model are here:
/data/envs/requirements.txt to create the environment:

python3 -m <name_of_env> ~/myenv
source ~/<name_of_env>/bin/activate
pip install -r /data/requirements.txt
pip list #to verify all the python packages
