# This script allows for submitting PySpark jobs in Google Dataproc
# Configuration paramenters can be modified according to needs

# Preprocessing (vocals isolation and segmentation) is performed
# using PySpark in the cloud with multiple workers for efficiency

bundle:
	python setup.py bdist_egg

# Config params
# ---- GCP params
GCP_PROJECT=alpha-xxxx
GCP_ROOT_BUCKET=preprocessing
GCP_ZONE=us-central1-a
GCP_MASTER_MACHINE_TYPE=n1-highmem-8
GCP_WORKER_MACHINE_TYPE=n1-highmem-8
GCP_MASTER_BOOT_DISK_SIZE=500
GCP_WORKER_BOOT_DISK_SIZE=500
GCP_IMAGE_VERSION=1.3-deb9
GCP_NUM_WORKERS=1
# pip packages to install in the Spark workers
PIP_PACKAGES=librosa==0.6.3 scipy==1.2.1 numpy==1.16.2 torchvision==0.2.2
# scripts to run during the nodes' initializations
INIT_ACTIONS=gs://$(GCP_ROOT_BUCKET)/init_cluster.sh\
,gs://$(GCP_ROOT_BUCKET)/my-pip-install.sh

# ---- Applicaton params
SAMPLING_RATE=44100
SIZE_DATASET=100
BATCH_SIZE=100
SIZE_FFT_CHROMA=1024
CONTEXT_LENGTH=10
DEBUG=0
CLUSTER_NAME=X


create-spark-cluster:
	gsutil cp init_cluster.sh gs://$(GCP_ROOT_BUCKET)/
	gsutil cp my-pip-install.sh gs://$(GCP_ROOT_BUCKET)/
	gcloud dataproc clusters create $(CLUSTER_NAME) --subnet default --zone $(GCP_ZONE) \
	--master-machine-type $(GCP_WORKER_MACHINE_TYPE) \
	--master-boot-disk-size $(GCP_MASTER_BOOT_DISK_SIZE) \
	--num-workers $(GCP_NUM_WORKERS) \
	--worker-machine-type $(GCP_WORKER_MACHINE_TYPE) \
	--worker-boot-disk-size $(GCP_WORKER_BOOT_DISK_SIZE) \
	--image-version $(GCP_IMAGE_VERSION) \
	--project $(GCP_PROJECT) \
	--metadata 'PIP_PACKAGES=$(PIP_PACKAGES)' \
	--initialization-actions $(INIT_ACTIONS)


run-preprocessing: bundle
	gsutil cp dist/preprocessing_spark-0.1-py3.5.egg gs://$(GCP_ROOT_BUCKET)/
	gsutil cp preprocess.py gs://$(GCP_ROOT_BUCKET)/
	gcloud dataproc jobs submit pyspark \
		--properties=spark.executor.cores=1,spark.driver.cores=8 \
		--cluster=$(CLUSTER_NAME) \
		--project $(GCP_PROJECT) \
		--py-files gs://$(GCP_ROOT_BUCKET)/preprocessing_spark-0.1-py3.5.egg \
					gs://$(GCP_ROOT_BUCKET)/preprocess.py \
		-- \
		--size-dataset $(SIZE_DATASET) \
		--sampling-rate $(SAMPLING_RATE) \
		--batch-size $(BATCH_SIZE) \
		--size-fft-chroma $(SIZE_FFT_CHROMA) \
		--context-length $(CONTEXT_LENGTH)