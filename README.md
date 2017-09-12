
# Image Superresolution in FloydHub

This document contains a guide on how to use FloydHub's platform to perform a deep learning task.

The task we will be handling in this guide is the superresolution of an image, using an efficient sub-pixel convolution layer described in  ["Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network" - Shi et al.](https://arxiv.org/abs/1609.05158).


<img src="./images/54082.jpg" width=250> <img src="./images/out_54082.jpg" width=450>

Left: input image / Right: output image with 3x upscale super-resolution after 30 epochs:

We will use this example to demonstrate how to train a model, evaluate its performance and finally deploy it to a REST API so that the model can be used through a HTTP request.


## Project setup

For this guide we will use [this](https://github.com/flsantos/floydhub-super-resolution) repository, forked from [Pytorch's super_resolution example](https://github.com/pytorch/examples/tree/master/super_resolution) and adjusted to run on FloydHub. Clone this repository to your local machine and sync it to a project in your FloydHub's account:

```
# Clone the repository to your local machine
$ git clone https://github.com/flsantos/floydhub-super-resolution

$ cd floydhub-super-resolution

# Sync this local repository to a project in your FloydHub's account
$ floyd init floydhub-super-resolution
```


## Prepare the data


##### The BSD300 dataset
The dataset for this task is the [BSD300 dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/), using crops from the 200 training images, and evaluating on crops of the 100 test images. This dataset is already uploaded and publicly available on [FloydHub](https://www.floydhub.com/fsantos1991/datasets/bsd300/1).

<img src="./images/dataset.jpg">


##### Use your own dataset
If you want to train and test the model with a different dataset, you can [create a dataset](http://docs.floydhub.com/guides/basics/create_new/#create-a-new-dataset) in FloydHub and upload it from your local machine.
```
# Create the dataset through Floyd's web dashboard

# Sync the local dataset directory to a remote dataset in your FloydHub's account
$ floyd data init imagenet-2017

# Start uploading...
$ floyd data upload


```

##### Mounting the dataset's directory
Once the dataset is in Floyd's platform, your code will have access to the dataset's directory by using the `--data` parameter of `floyd run` command. 

For example, when running `floyd run --data fsantos1991/datasets/bsd300/1:my_dataset_dir "python hello_world.py"`, the `--data` parameter makes sure that the dataset will be available at `/my_dataset_dir` directory.


## Train the model

You can train the superresolution model by running the `main.py` with the required parameters. Below is the [run](http://docs.floydhub.com/commands/run) command to start training the model on Floyd:

```
floyd run --env pytorch \
          --data fsantos1991/datasets/bsd300/1:input \
          --gpu \
          "python main.py --upscale_factor 6 \
                          --batchSize 4 \
                          --testBatchSize 100 \
                          --nEpochs 50 \
                          --lr 0.001 \
                          --cuda \
                          --trainPath /input/images/train \
                          --testPath /input/images/test \
                          --outputPath /output"
```
Notes:

* This job will run on a pytorch environment `--env flag`
* The `/input` directory was mounted with the [fsantos1991/datasets/bsd300/1](https://www.floydhub.com/fsantos1991/datasets/bsd300/1) dataset.
* This job will run on a machine with GPU `--gpu flag`
* Program parameters `--trainPath` and `--testPath` are directories from the mounted dataset
* Program parameter `--outputPath` defines where to save the model checkpoints
* Train and test datasets are passed because the script iteractively evaluates the test dataset after every epoch

Once the job is launched, you can follow along the progress by using the [logs](http://docs.floydhub.com/commands/logs) command.
```
$ floyd logs <JOB_NAME> -t
```

Floyd saves any content stored in the /output directory after the job is finished. This output can be used as a datasource in the next project. To get the name of the output generated by your job use the [info](http://docs.floydhub.com/commands/info) command.
```
$ floyd info <JOB_NAME>
```






## Evaluate the model


##### Evaluate a test dataset

You can evaluate your model's performance by running the `evaluation.py` script with the required parameters.
```
floyd run --env pytorch \
          --data fsantos1991/datasets/bsd300/1:input \
          --data fsantos1991/projects/floydhub-super-resolution/11/output:models \
          --gpu \
          "python evaluation.py --model /models/model_epoch_30.pth \
                                --upscale_factor 3 \
                                --testBatchSize 100 \
                                --cuda \
                                --testPath /input/images/test"
```
Notes:

* The `--data` parameter is used twice in order to mount the dataset in `/input` and the output from the training job in `/models`
* Program parameter `--model` is a file path from the mounted output
* Program parameter `--testPath` is a directory from the mounted dataset


You can track the status of the job with the status or logs command.
```
$ floyd status <JOB_NAME>
$ floyd logs <JOB_NAME> -t
``` 


##### Evaluate a single image file

You can evaluate your model's performance by running the `super_resolution.py` script to super resolve a single image and store its output image.

```
floyd run --env pytorch \
          --data fsantos1991/datasets/bsd300/1:input \
          --data fsantos1991/projects/floydhub-super-resolution/11/output:models \
          --gpu 
          "python super_resolve.py --input_image /input/images/test/16077.jpg \
                                   --model /models/model_epoch_30.pth \
                                   --output_filename /output/out.png \
                                   --cuda"
```
Notes:

* The `--data` parameter is used twice in order to mount the dataset in `/input` and the output from the training job in `/models`
* Program parameter `--model` is a file path from the mounted output
* Program parameter `--output_filename` is a target file path where the job output will be available

You can view the saved output of a job using the floyd output command:
```
$ floyd output fsantos1991/projects/floydhub-super-resolution/1
Opening output directory in your browser...
```


## Serve the model through a REST API

FloydHub supports a serving mode for demo and testing purposes. If you run a job with `--mode serve` flag, FloydHub will run the app.py file in your project and attach it to a dynamic service endpoint.

##### Serve mode
When the `--mode serve` flag is used, it will upload the files in the current directory and run a special command - `python app.py`. Floyd expects this file to contain the code to run a web server and listen on port 5000. You can see the [app.py](https://github.com/flsantos/floydhub-super-resolution/blob/master/app.py) file in the same repository. This file handles the incoming request, executes the code in super_resolve.py and returns the output image.
```
floyd run --env pytorch \
          --data fsantos1991/projects/floydhub-super-resolution/11/output:models \
          --mode serve \
          --gpu
```
Notes

* If your app.py script requires additional Python packages, you will need to add these [dependencies](http://docs.floydhub.com/guides/jobs/installing_dependencies/) to a file called `floyd_requirements.txt`.

##### Request the API

Once your serving job is up and running, you can send any image file as request to this api and it will return the super resolved image.
```
curl -o ./images/out_54082.jpg -F "file=@./images/54082.jpg" -F "model=model_epoch_30.pth" https://www.floydhub.com/expose/Lq6FZzPUQLp9R8ingzEmtG
```

