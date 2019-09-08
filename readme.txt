#### NOTE: --reservation keeps changing time to time on cluster, if you are unable to submit the batch job, recheck the reservation ids using following command
> scontrol show reservation

directory structure ~~
|__/dataset           --> Contains all audio .wav file and transcript .txt files
|__/requirements.txt  --> Contains the information of all the software packages to be installed
|__/build_data_csv.py --> Builds a csv from the dataset with paths and transcripts 
|__/build_vocab.py    --> Builds the linguistic language character level vocabulary for hindi.
|__/ds2_train.sh      --> Used to run the training code on slurm scheduler.
|__/ds2_infer.sh      --> Used to run the inference code on slurm scheduler.
|__/OpenSeq2SeqInfer  --> contains trained model and run.py that will be called from ds2_infer.sh for testing
   |__/run.py         --> Code run for training and inference
   |__/open_seq2seq/hindi_speech_data/ --> Directory for the dataset csv created by build_data_csv.txt
      |__/hindi_train.csv              --> Partition of the data on which training was done
      |__/hindi_val.csv                --> Data on which validation is performed
      |__/hindi_test.csv               --> Data for testing the model
   |__/ds2_log                         --> Folder containing the log files and model checkpoints
   |__/ds2_small_config.py             --> Contains the model layers and network architecture


### for python2.7 and the dependencies
> pip install requirements.txt


Step 1: Create data.csv file for feeding the path of wav files and corresponding transcript to the model
> python build_data_csv.py ./dataset

### ./dataset is the location of the folder containing wav and txt files of the dataset



Step 2: Build vocabulary of characters from the dataset, to be used as the target to the model
> python build_vocab.py ./dataset

###./dataset is the location of the folder containing wav and txt files of the dataset


--------------For training DS2Small------------
Step 3: Submit the job onto slurm scheduler using following command

> sbatch --reservation=hackathon2019_73 ds2_train.sh         --> Shell script to intiate the model training

### This will start training if no log directory is found otherwise it will throw an ERROR "include --continue_learning flag" if model already exists
### The logs will be saved in the directory "~/OpenSeq2SeqInfer/ds2_log/"
### If you wish to fine tune your model over already saved checkpoints then submit include "--continue_learning" flag

### ds2_train.sh files execute "~/OpenSeq2Seq/run.py" with "--mode=train" internally
### Train, evaluation and test files are stored at location "~/OpenSeq2Seq/open_seq2seq/hindi_speech_data"
### To look or edit Vocabulary File, log directory, data files(train, val and test) directory edit config file 
### -which is present at "~/OpenSeq2Seq/ds2_small_config.py"
 


---------------For inference/testing a new wav file -----------
Step 4: Steps to run test the new audio file and check the WER(Word error rate).

### Command to create the csv file from wav file#################
> python build_infer_csv.py <path to test dataset directory>

### eg: python build_infer_csv.py ./temp/small_data/


### (Optional:) To see the created csv and see the content of the file:
> cat OpenSeq2SeqInfer/open_seq2seq/hindi_speech_data/hindi_sample.csv

### To convert wav file to hindi text use below command, it schedules the job for conversion of wav files to text on GPU cluster. i.e. Submit the job on the slurm scheduler using the following command.

> sbatch --reservation=hackathon2019_73 ds2_infer.sh 

### To see the job submitted to the slurm scheduler run following command

> squeue

### ds2_infer.sh files execute "~/OpenSeq2Seq/run.py" with "--mode=infer" internally
### Output will be stored in ./OpenSeq2SeqInfer/file.txt

> cat ./OpenSeq2SeqInfer/file.txt

### To find the Average Word Error Rate(WER) on the test set. (Only if .txt files are provided along with .wav files)

> python eval_model.py


