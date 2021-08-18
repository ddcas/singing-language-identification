

# Singing-language Identification

![](https://github.com/ddcas/singing-language-identification/blob/main/img/slid.PNG?raw=true)



##Introduction
This repository corresponds to a MSc thesis project on Singing-language Identification (SLID).
SLID is a research problem within Music Information Retrieval that consists in identifying the correct language of the vocals sung in an arbitrary music recording.


##Preprocessing
Since language is a relatively high-level feature in musical data, the SLID problem requires large amounts of music recordings for each of the target languages. Additionally, the variety of music styles and genres makes the need for lots of datapoints critical, due to the high variance of the dataset distribution.

However, even with a large dataset, the features that best characterize languages in music might be elusive, still for an expressive enough classifier. There is where preprocessing increases the data efficiency, by filtering out the useless information from each music recording and condensing the crucial features in smaller vectors.
#####1) Vocals isolation
The isolation of the vocals consists in separating the vocals from the background music in the music recording and keeping them as the main source of language information.
It is based on the assumption that audio-linguistic features are more useful for language identification than audio-instrumental features. For example: there are many Japanese rock songs in the dataset, but there are not the only songs with background rock music.

A neural-network-based [source-separation algorithm]((https://github.com/dr-costas/mad-twinnet)) was applied to all music recordings, followed by a removal of the non-vocal sections using `librosa.effects.split`.

#####2) Segmentation
Once the vocal segments are isolated, a combination of algorithms is employed to find those vocal pieces that are more likely to contain the language-characteristic features:  `librosa.feature.chroma_stft` --> [autocorrelation](https://en.wikipedia.org/wiki/Autocorrelation) --> `librosa.feature.rms` --> `librosa.feature.spectral_flatness`.


 **PySpark preprocessing**
Due to the size of the dataset and the nature of audio waveform data, the preprocessing was performed using PySpark on Google Cloud Dataproc for efficiency purposes.
A `Makefile` is included to facilitate this process.
1. Create the Dataproc cluster:  `make create-cluster CLUSTER_NAME=<cluster_name> SIZE_DATASET=<size_dataset> ...`
2. Submit PySpark job: `make run-preprocessing CLUSTER_NAME=<cluster_name> SIZE_DATASET=<size_dataset> ...`


**Requirements**
          `librosa == 0.6.3`
          `scipy == 1.2.1`
          `numpy == 1.16.2`
          `torchvision == 0.2.2`
          `Soundfile == 0.10.2`
          `tensorflow == 1.14`

##Training
The resulting outputs of the preprocessing should be waveforms containing vocal segments. These will serve as inputs to the classifier that will try to predict the language of the vocals contained in each segment.

The classifier chosen is a [Temporal Convolutional Network](https://github.com/locuslab/TCN) (TCN), which has demonstrated to be effective across a diverse range of sequence modeling tasks.

 **AI Platform training**
Due to the size of the dataset and the nature of audio waveform data, the training, validation and inference was performed on Google Cloud AI Platform for efficiency purposes.
A `Makefile` is includede to facilitate this process.
Submit PySpark job: `make run-training JOB_NAME=<job_name> SIZE_DATASET=<size_dataset> ...`


**Requirements**
  `tensorflow == 1.14`
`torchvision == 0.2.2`
`matplotlib == 2.2.4`
      `stacklogging == 0.1.2`
      `Soundfile == 0.10.2`


