this is how you are going to deal with this code first of all you need to have the droid cam app downloaded
the "opening_cam_for_testing.py" file is for you try out if your phone camera is working and test it out.
if you don't have a camera that's what the code is oriented to but if you want to use your own camera which would give better results 
you need to adjust the code for that.
well now that you have the droid cam app downloaded on both your phone and your computer you are ready.
first of all you are supposed to collect the data and that is by using the "data_collector.py" file.
after you have inserted your phone ip and choose which folder or emotion it is going to take pics of in the code,
the camera would open and it would start taking pictures of your face and tell you how much it took so far
so you need to be mindfull of the folowing to get the best results:
1. Make sure you make the same exact impression or pose in the pic you want,
most importantly (use the same hand and do it in the same side of your face "so if you are closing an eye always close the same eye and dont switch between left and right").
2. Make sure you take pics in different light exposure so change the lights and their placements in the pics.
3. Make sure you are not doing something too similar to another label because it will confuse the model and might lead to higher false positives.
4. Make sure you change the camera's postion relative to your face and the other way around between pics.
5. Don't leave your hair in the same position in all the pics make sure you mess it up every now and then and change it so the model doesn't learn based on your hair.
6. Don't stay in the same state for too long the more diverse the data/pics are the better.
7. Don't move your head too aggressivly because it will create bulrry images.
8. Make sure the background isn't causing trouble by collecting pics of things that aren't your face because it will be trouble in both you having to delete it manually later
and it might recognize those objects as faces later in the detection phase and leading to lower accuracy in reading your face.
9. The higher the amount of images the better mine were 2000 images for each label.
10. Make sure all the labels have a similar amount of images in them so no bias is created with 8% difference allowed.

Now that you have your first batch of data you should start deleting some of it in the "data_deleter.py" file just adjust how much you want to delete by changing the modulos option
and selecting the file you want to delete from so that you don't have a lot of duplicate images.
also go and delete some images yourself that are (not of your face,blurry,wrong pose,look duplicated or too much of them)
If you want to make more images make a folder named temp and go back to the data collection and then move them back "ctrl + x" and "ctrl + v" to their desired area
don't worry about them having a wrong label because this will be solved in the next step.

The next step being the file "relabler.py" since if you delete the images it will cause a gap ("image_1,image_3" that is becasue you might have deleted "image_2")
or maybe because you added more images from the temporary folder that would hold the name of that folder.
so you just select the folder name and it will label everything correctly 

Now that all the data is good to go and they are in the "dataset" folder you use the "compress_dataset.py" so you can put it in google colab to train the model

DON'T run the "google_colab_code.py" on your local computer you should run it on google colab:https://colab.research.google.com/
and select a gpu in the runtime so you get to train your model fast
make sure to run each cell the way i hilighted in the file and download it in the same place as the rest of the code
put the model and the rest of the info in the "meme_detector.py" file and congratiulations you can enjoy the experince.

