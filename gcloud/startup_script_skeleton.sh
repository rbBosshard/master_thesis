sudo apt-get update && sudo apt-get install -y git && sudo apt-get install -y --fix-missing python3-pip
sudo git config --global pull.ff only
sudo git clone https://[GITHUB_TOKEN]@github.com/rbBosshard/pytcpl.git
sudo pip install -r pytcpl/requirements.txt