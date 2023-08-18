# Foldy-in-a-box

This Foldy deployment option is a quick setup on a Google Cloud Engine device with a GPU. It is a fully functional Foldy instance which can be started very quickly, and still customized.

## Deployment

The "five minute deployment" described below includes steps to reserve a Google Cloud Virtual Machine, then a one-liner to download and setup Foldy. It will take 24-48 hours after that moment for the full AlphaFold databases to finish downloading, but you can log in and use the Foldy frontend in the meantime.

Following the "five minute deployment" are high recommended security steps, which do not constitute a guarantee of security, but rather offer a few minimal sources of defense for your computational resource.

1. **Using the Cloud Console**
    1. Create a GCE instance with a GPU
        * Sign into your Google Cloud Project in the Google Cloud Console, and go to `Compute Engine > VM instances`
        * Select `Create Instance` [[direct link](https://console.cloud.google.com/compute/instancesAdd)]
        * Name the instance something memorable like `foldybox`
        * Choose a zone which has the appropriate GPUs available (eg, at time of writing region `us-central1` zone `us-central1-a` has T4 GPUs available)
        * Under `Machine Configuration`
            * Choose the GPU you want. We recommend 1 T4 for handling smaller proteins / complexes and A100 80GB for the biggest complexes.
            * For Machine Type choose a large base, we recommend the `High memory` option `n1-highmem-8`
        * Under `Boot Disk` it probably suggests switching to an image which better supports GPUs such as `Deep Learning VM with CUDA 11.3 M110`. Click "Switch", then also change the size of the boot disk to 3000GB, to support installing the AlphaFold databases and holding your fold outputs.
        * Under `Firewall` select "Allow HTTP traffic".
    2. Install Foldy
        * You can either read and mimic the setup steps in `install_foldy_in_a_box.sh`, or you can use the console to open an SSH connection to the instance and call:
            ```bash
            wget -O - https://raw.githubusercontent.com/JBEI/foldy/main/deployment/foldy-in-a-box/install_foldy_in_a_box.sh | bash
            ```
        * It will take about 30 minutes for the web interface to become available, and 48 hours for the databases to finish downloading. Make sure to leave the instance running without interruption for the databases to finish downloading. You can view the progress of the installation by SSHing into the instance and calling `sudo journalctl -u foldy.service -f`.
    3. You can now access Foldy from the external IP address listed next to the instance in the Google Cloud console. You can put the IP address listed into your browser like `http://{IP_ADDRESS}`. Make sure you use `http` not `https`.
2. **Using the gcloud command line**
    1. Create the instance. You can change this command to fit your needs, as written it creates a machine called foldybox and allocates an Nvidia T4 GPU:
          ```bash
          gcloud compute instances create foldybox \
          --zone=us-central1-a \
          --machine-type=n1-highmem-8 \
          --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \
          --maintenance-policy=TERMINATE \
          --provisioning-model=STANDARD \
          --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append \
        --accelerator=count=1,type=nvidia-tesla-t4 \
        --tags=http-server \
        --create-disk=auto-delete=yes,boot=yes,device-name=instance-1,image=projects/ml-images/global/images/c0-deeplearning-common-cu113-v20230807-debian-10,mode=rw,size=3000,type=pd-balanced \
        --no-shielded-secure-boot \
        --shielded-vtpm \
        --shielded-integrity-monitoring \
        --labels=goog-ec-src=vm_add-gcloud \
        --reservation-affinity=any
        ```
        * It may take a moment before you can SSH in.
    2. Install Foldy
        * First, SSH in. If the name of your instance is foldybox and it is located in us-central1-a, you can run:
            ```bash
            gcloud compute --zone=us-central1-a ssh
            ```
        * If it asks to install NVIDIA drivers, say yes.
        * You can either read and mimic the setup steps in `install_foldy_in_a_box.sh`, or you can use the console to open an SSH connection to the instance and call:
            ```bash
            wget -O - https://raw.githubusercontent.com/JBEI/foldy/main/deployment/foldy-in-a-box/install_foldy_in_a_box.sh | bash
            ```
        * It will take about 30 minutes for the web interface to become available, and 48 hours for the databases to finish downloading. Make sure to leave the instance running without interruption for the databases to finish downloading. You can view the progress of the installation by SSHing into the instance and calling `sudo journalctl -u foldy.service -f`.
    3. You can now access Foldy from the external IP address listed next to the instance in the Google Cloud console. You can put the IP address listed into your browser like `http://{IP_ADDRESS}`. Make sure you use `http` not `https`.


## Highly Recommended Changes

* **Allow-list access to your machine.** Your Foldy instance is now publicly available to the internet, and bad actors will quickly connect. A surprisingly strong protection against attacks on the open port is to use an "IP allow-list," which restricts inbound connections to a short list of machines. You can specifically list your home network or office computer, but the best use of an IP allow-list is to allow only IPs from your organization's VPN, which will allow you to connect to the instance while on VPN. Most organizational VPNs route traffic through a block of IP addresses which can be concisely specified in [CIDR notation](https://en.wikipedia.org/wiki/Classless_Inter-Domain_Routing). To allow-list a specific IP or CIDR block to access your Foldy instance:
  * Open your instance in the Google Cloud console (don't click Edit)
  * Scroll down to "Network interfaces" and open the one network interface (maybe named "nic0")
  * Under "Firewall and routes details" open "vpc-firewall-rules" and select the http rule (maybe named "default-allow-http")
  * Hit "Edit"
  * Specify the "Source IPv4 ranges" to either be a list of IP addresses to allow-list, or a block of IP addresses in CIDR notation
  * Save
* **Switch to a static external IP address.** By default your machine's IP address is ephemeral, meaning google may change it any time. You can reserve a static external IP address and assign it to your machine by opening your instance from the "VM Instances" page and hitting "Edit". Under `Network interfaces` select the one listed network interface, likely called `default`. Change `External IPv4 address` to a static external IP address, which you can reserve from the dropdown.
