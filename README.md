# ImageNet training in PyTorch

## About

This Ai uses x-ray images to dicern whether or not someone has pneumonia! In this project we will be retraining an image classification model.
btw my export to github was weird and didnt push my code >:( so you'll be able to find it near the bottom 


## Getting started 

To start head to: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia 
download and unzip the zip file (This will most likley take a while so dont worry to much)

Scp this dataset into your nano now, if you forgot the command will look smth like this: scp C:\Users\computer\username\desktop\foldername.. user_name@ip_adress:/home/Nvidia_username/wherever you want it 

-note if your on windows you might need to add onedrive or something like that. You can figure this out by going into your folder tapping the folder icon and it should look smth like this 

C:\Users\qinzh\Desktop\Example_Folder

I personally have everything in jetson_inference/python/training/classification/data

you will need to have intalled pytorch, if you are missing anything use the pip install command 


## Quick note: 
Idk why but when I pushed all of the stuff from my project to the github none of the stuff I actually made got over here some go over it here in the read me 

## Right Before Training 
Make sure when you open your folder (in powershell) and make sure that it has the 4 following parts labels.txt, test, train, and val 
 - if you dont have labels.txt type nano labels.txt and most likley your labels will be PNEUMONIA and NORMAL 
 - if you have any extra things type mv directory_name .. and it should move it out into the previous directory 
 
 If you've done what i have and put it in classification you should be able to access train.py. 
 anyways go back into jetson_inference and type out: ./docker/run.sh 
 your next line should start with something like root@root/Users: or something like this (i forgot what it actually says but it should say root somewhere) 
 go back into classification from here and we can start training 
 
## Training 
Now in classification we'll run the command python3 train.py --model-dir=models/model_name --batch-size=4 --workers=1 epochs=30 data/chest_xray(or whatever you named your dataset) when we set batch size to 4 and workers to one we're basically just trying to save a bit  of space 
epochs is basically howmany times we wanna train our model. a good amount is usually 30 ish but if you just wanna try you may set it to 1, just keep in mind your model will get more accurate the more you train it (dont go oveboard and train it like 100 times though :||| )

## Exporting 
Now that our model is done being trained we need to export it to be used 
python3 onnx_export.py --model-dir=models/whatever_you_named_it - this is the command

## Using our model to classify some images! 
 now that we've exported our model we need to use it 
were gonna make a quick shortcut, so type:
-NET=models/whatever_its_named
-DATASET=data/whatever_its_named

and now to actually use all of this type (or copy and paste then change lol) - it should sort of work like this if it dont try to command under this one (it will do all of the pictures though)
imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/find_a_picture

and this should classify one of your images!

If you want to classify all of them first make a directory for them to go to (mkdir normal)
then type (and modify): imagenet --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/../labels.txt \$DATASET/test/NORMAL $DATASET/normal

## Your amazing! 
This should be completed and ready to go! you can input images and see whether or not your patient has Pneumonia but if you wanna get creative and create a script to do this automatically you can, i have an example written down, you'll need to figure out how to get your model in but it should look something like this:

import jetson.inference
import jetson.utils

from jetson.inference import imageNet

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="filename of the image to process")
parser.add_argument("--network", type=str, default="googlenet", help="model to use, can be:  googlenet, resnet-18, ect. (see --help for others)")
opt = parser.parse_args()


img = jetson.utils.loadImage(opt.filename)

net = jetson.inference.imageNet(opt.network)
#i was also experimenting adding in our network using net = imageNet(model="wherever your model is ex: models/Pneumonia/resnet18.onnx", labels="wherever your labels # are located", input_blob="input_0", ouput_blob="output_0")

class_idx, confidence = net.Classify(img)

class_desc = net.GetClassDesc(class_idx)

if(class_desc == "pneumonia"):

  print("U have Pneumonia...")
  
else:

  print("you dont have Pneumonia")
  
  
 ## Have a nice day 









