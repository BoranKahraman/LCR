import os
import subprocess
import time

input_base = "/Users/nilakay/Desktop/quoters/quoters_15"
output_base = "/Users/nilakay/Desktop/xmls/15"
grobid_client_path = "/Users/nilakay/grobid_client_python"
docker_image = "lfoppiano/grobid:0.7.1"
log_file = "/Users/nilakay/Desktop/processed_folders15.log"

#function to check if any Grobid container is running
def is_docker_running():
    result = subprocess.run(["docker", "ps", "--filter", "ancestor=" + docker_image, "--format", "{{.ID}}"], capture_output=True, text=True)
    return bool(result.stdout.strip())

#function to start a new Docker container if none is running
def ensure_docker_running():
    if not is_docker_running():
        print("Starting a new Docker container for Grobid...")
        subprocess.run([
            "docker", "run", "-d", "--rm", "--init", "-p", "8070:8070", docker_image
        ])
        time.sleep(10)  #wait for the container to fully start

#function to read the log file and get the list of processed folders
def get_processed_folders(log_file):
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            return set(line.strip() for line in f)
    return set()

#function to log a processed folder
def log_processed_folder(log_file, folder_name):
    with open(log_file, 'a') as f:
        f.write(folder_name + '\n')

#ensure the output base directory exists
os.makedirs(output_base, exist_ok=True)

#change to the directory where grobid_client is located
os.chdir(grobid_client_path)

#get the list of already processed folders
processed_folders = get_processed_folders(log_file)

#loop through each folder in the input base directory
for folder_name in os.listdir(input_base):
    input_folder = os.path.join(input_base, folder_name)
    output_folder = os.path.join(output_base, folder_name)

    if os.path.isdir(input_folder) and folder_name not in processed_folders:
        #ensure the output folder exists
        os.makedirs(output_folder, exist_ok=True)

        #ensure Docker container is running
        ensure_docker_running()

        #command to run the Grobid client
        command = [
            "python3", "-m", "grobid_client.grobid_client",
            "--input", input_folder,
            "--output", output_folder,
            "--verbose", "processFulltextDocument"
        ]

        #run the Grobid client for each folder
        print(f"Processing {input_folder}...")
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error processing {input_folder}: {result.stderr}")
        else:
            print(f"Successfully processed {input_folder}")
            #log the processed folder
            log_processed_folder(log_file, folder_name)

print("Processing complete.")
