# Foldy-in-a-box

This Foldy deployment option is a quick setup on a Google Cloud Engine device with a GPU. It is a fully functional Foldy instance which can be started very quickly, and still customized. We recommend this option for individual labs.

## Deployment

The "five minute deployment" described below includes steps to reserve a Google Cloud Virtual Machine, then a one-liner to download and setup Foldy. It will take 24-48 hours after that moment for the full AlphaFold databases to finish downloading, but you can log in and use the Foldy frontend in the meantime.

Following the "five minute deployment" are high recommended security steps, which do not constitute a guarantee of security, but rather offer a few minimal sources of defense for your computational resource.

Instructions are available using the [Cloud Console](#installation-through-the-cloud-console) or the [GCloud Command Line tool](#installation-using-the-gcloud-command-line-tool)

### Requirements
1. A Google Cloud Project
2. A login for the project which includes permissions to create GCP instances

### Installation Through the Cloud Console
Go to the [Google Cloud Console](https://console.cloud.google.com/welcome) and log into your Google Cloud project.

**Steps**
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
    * That's it, you can now create your instance. **If instance creation fails, check the [debugging](#debugging) steps below.**
      > Note that resource availability errors are common. The demand for GPUs is very high these days. \ varies by GPU type, by zone, and even by time of day. We recommend trying other resource types (eg, change to a different size GPU), try a different zone, or try again later. See [Debugging](#debugging) for more info.

2. Install Foldy
    * Once the machine is started, you can use the Cloud Console to SSH from your browser. Look for an "SSH" button on the row next to your instance. You can also SSH using the gcloud command line tool. See instructions below.
    * If it asks to install NVIDIA drivers, say yes.
    * You can either read and mimic the setup steps in `install_foldy_in_a_box.sh`, or you can use the console to open an SSH connection to the instance and call:
        ```bash
        wget -O - https://raw.githubusercontent.com/JBEI/foldy/main/deployment/foldy-in-a-box/install_foldy_in_a_box.sh | bash
        ```
    * Once this command completes, you're all set! It will take about 20-30 minutes to build all the code before the web interface is available, and up to 48 hours for the databases to finish downloading. Make sure to leave the instance running without interruption for the databases to finish downloading. You can view the progress of the installation by SSHing into the instance and calling `sudo journalctl -u foldy.service -f`.
    * *Note that sometimes the databases fail to download.* You can check status with the journalctl command above, or just wait to see if your first jobs succeed. If they don't succeed, you can prompt the databases to re-download by restarting the instance, or SSHing into the instance and restarting the Foldy service with `sudo systemctl restart foldy.service`.
3. You can now access Foldy from the external IP address listed next to the instance in the Google Cloud console. You can put the IP address listed into your browser like `http://{IP_ADDRESS}`. Make sure you use `http` not `https`.

### Installation Using the GCloud Command Line Tool
Make sure you have [installed](https://cloud.google.com/sdk/docs/install-sdk) and logged into the GCloud command line tool. Verify you have set the proper project with `gcloud config get project`. You can find the GCloud CLI cheat sheet [here](https://cloud.google.com/sdk/docs/cheatsheet).

**Steps**
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
      > Note that resource availability errors are common. The demand for GPUs is very high these days. Supply varies by GPU type, by zone, and even by time of day. We recommend trying other resource types (eg, change to a different size GPU), try a different zone, or try again later. See [Debugging](#debugging) for more info.
    * It may take a moment before you can SSH in. **If instance creation fails, check the [debugging](#debugging) steps below.**
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
    * Once this command completes, you're all set! It will take about 20-30 minutes to build all the code before the web interface is available, and up to 48 hours for the databases to finish downloading. Make sure to leave the instance running without interruption for the databases to finish downloading. You can view the progress of the installation by SSHing into the instance and calling `sudo journalctl -u foldy.service -f`.
    * *Note that sometimes the databases fail to download.* You can check status with the journalctl command above, or just wait to see if your first jobs succeed. If they don't succeed, you can prompt the databases to re-download by restarting the instance, or SSHing into the instance and restarting the Foldy service with `sudo systemctl restart foldy.service`.
3. You can now access Foldy from the external IP address listed next to the instance in the Google Cloud console. You can put the IP address listed into your browser like `http://{IP_ADDRESS}`. Make sure you use `http` not `https`.


## Debugging

Instance creation can fail for a few common reasons:
* **A resource is unavailble.** Google cloud supply and demand varies constantly. Sometimes certain resources (eg, "n1-highmem-8" machines or "Nvidia T4" GPUs) are unavailable when you try to create your instance. Unfortunately, it is not easy to see which zone has availablity for any given resource. Instead, you should retry at another time, or retry in another zone. Eg, if creating your instance in "us-central1-a" fails due to resource availability, you can try creating your VM in "us-central1-f". You can GPU availability by zone [here](https://cloud.google.com/compute/docs/gpus/gpu-regions-zones), and you can see GPU performance [here](https://cloud.google.com/compute/docs/gpus/#performance_comparison_chart). Some GPUs that we've tested:
  * A100 (40GB or 80GB): These are the resources which DeepMind uses to calculate their largest folds, and we have used A100s w/ 80GB memory to predict structures up to 6000 amino acids.
  * T4: We have used this as a more affordable deployment option, and have predicted structures up to 1000 amino acids.
  * Others: Although we haven't tested other GPU types, we think other types should work as well. It seems that total memory is the determining factor for max structure size, and that FLOPS determines the speed of the prediction.
* **Insufficient Quota.** Every project has many limits imposed on the resources it can use. If you created a new Google Cloud project, and it's not associated with an institution, your limits will likely start quite low. You can request an increase in your quota through [quota page](https://console.cloud.google.com/iam-admin/quotas). For instance, when installing Foldy in a new project, you'll likely run into limits for both the `Persistent Disk SSD` and `GPUS-ALL-REGIONS-per-project` quotas, whose defaults are something like 500GB and 0, respectively. Note that the `Persistent Disk SSD` quota is per-*region*, so you need to increase the quota for the appropriate region. Eg, if you're making your Foldy instance in region `us-central1` and zone `us-central1-a`, then you need to request more quota for region `us-central1`.

## Highly Recommended Changes

Your Foldy instance is now publicly available to the internet, and bad actors will quickly connect. Your security is your responsibility, but we offer some suggestions below for securing your instance. We also suggest asking your institution's IT department for advice and institutional best practices. You can read more about securely accessing Google Cloud resources [here](https://cloud.google.com/solutions/connecting-securely).

* **Use an SSH-tunnel for access.** One solution which is quite secure though fairly cumbersome for end-users is for all users to SSH-tunnel into the instance. In this solution, the first step is to **disable HTTP connections to your instance.** Now every Foldy user must have a Google Cloud account and be a member of the Google Cloud project with appropriate permissions. To connect to Foldy they'll use the gcloud command to create a "tunnel" between their computer and the google cloud machine. For instance, if the VM name is "foldybox", the command to create an SSH tunnel might look like `gcloud compute ssh -- -NL 80:localhost:80`. You'll also have to change the value of `MY_URL=localhost:80` in `/foldy/deployment/foldy-in-a-box/startup.sh` and `/foldy/deployment/foldy-in-a-box/prestartup.sh` for routing within the app to work correctly. Once the tunnel is open, users can use Foldy in their browser by navigating to `http://localhost:80`.
* **Allow-list access to your machine.**  A surprisingly strong protection against attacks on the open port is to use an "IP allow-list," which restricts inbound connections to a short list of machines. You can specifically list your home network or office computer, but the best use of an IP allow-list is to allow only IPs from your organization's VPN, which will allow you to connect to the instance while on VPN. Most organizational VPNs route traffic through a block of IP addresses which can be concisely specified in [CIDR notation](https://en.wikipedia.org/wiki/Classless_Inter-Domain_Routing). To allow-list a specific IP or CIDR block to access your Foldy instance:
  * Open your instance in the Google Cloud console (don't click Edit)
  * Scroll down to "Network interfaces" and open the one network interface (maybe named "nic0")
  * Under "Firewall and routes details" open "vpc-firewall-rules" and select the http rule (maybe named "default-allow-http")
  * Hit "Edit"
  * Specify the "Source IPv4 ranges" to either be a list of IP addresses to allow-list, or a block of IP addresses in CIDR notation
  * Save
* **Switch to a static external IP address.** By default your machine's IP address is ephemeral, meaning google may change it any time. You can reserve a static external IP address and assign it to your machine by opening your instance from the "VM Instances" page and hitting "Edit". Under `Network interfaces` select the one listed network interface, likely called `default`. Change `External IPv4 address` to a static external IP address, which you can reserve from the dropdown.
