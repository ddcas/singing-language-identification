# This script allows for submitting AI Platform jobs in Google Cloud
# Configuration paramenters can be modified according to needs

# Training, validation and testing is performed in the cloud with
# multiple workers equipped with GPUs for efficiency


# Config params
# ---- GCP params
GCP_PROJECT=alpha-xxxx
GCP_REGION?=us-central1
GCP_JOB_ID=`date "+%Y%m%d%H%M%S"`
GCP_JOB_NAME?=training_$(JOB_ID)
GCP_ROOT_BUCKET?=training
GCP_JOB_DIR?=gs://$(ROOT_BUCKET)/outputs
GCP_DATA_DIR?=gs://$(ROOT_BUCKET)/data
GCP_LOG_DIR?=$(JOB_DIR)/logs/

# ---- Applicaton params
SIZE_DATASET=800
LEN_SEQ=1024
SIZE_KERNEL=3
SIZE_BATCH=100
NUM_LAYERS=8
NUM_HIDDEN=150
EPOCHS=50
DEBUG=0
SR=44100
NUM_FILES=2500
DROP_OUT=0.25
LEARNING_RATE=0.001


run-training:
	gcloud ai-platform jobs submit training \
	$(JOB_NAME) \
	--project $(GCP_PROJECT) \
	--region $(GCP_REGION) \
	--package-path trainer \
	--config trainer/config.yaml \
	--module-name trainer.train \
	--job-dir $(JOB_DIR) \
	-- \
	--sr $(SR) \
	--job-id $(GCP_JOB_ID) \
	--data-dir $(GCP_DATA_DIR) \
	--epochs $(EPOCHS) \
	--size-batch $(SIZE_BATCH) \
	--num-layers $(NUM_LAYERS) \
	--num-hidden $(NUM_HIDDEN) \
	--size-dataset $(SIZE_DATASET) \
	--drop_out $(DROP_OUT) \
	--lr $(LEARNING_RATE) \
	--len-seq $(LEN_SEQ)
