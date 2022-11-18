# PCPosture

A program to warn you when you slouch! 
![demo image](https://user-images.githubusercontent.com/46868596/202573176-b1257e3f-d8f6-4dd7-b5b4-08228781da6e.png)

Originally created as a school project, this was later submitted for the 2022 Congressional App Challenge.<br>

Watch the submission video for a quick demo and explanation [here](https://www.youtube.com/watch?v=fhw6DTA7wDw)!

## How Does it work?
1. A video feed is taken in from a camera positioned on the user's desk. 
2. Code from [StridedTransformer3D](https://github.com/Vegetebird/StridedTransformer-Pose3D) is used to calculate the user's pose in 2D space. 
3. To increase accuracy from differing angles, code from [EvoSkeleton](https://github.com/Nicholasli1995/EvoSkeleton/) is used to estimate the locations within 3D space. 
4. Keypoints from the lower half of the body are removed, as they cannot accurately be assessed.
5. The remaining keypoints are fed into a neural network which determines whether the posture is good or bad.
6. A point is to the user's score if they have good posture, while point is taken away if they have bad posture.
7. Upon reaching a user-defined lower threshold (this prevents a warning for one faulty reading), a signal is sent to the Arduino to give the user a phyiscal reminder.

## How was it trained?
This was trained on my laptop. I wrote a custom data generator to accomodate my data (10x3 arrays of 3D points). The structure is 3-layer MLP with 12 units in each layer, and a softmax binary output. Definitely exceedingly simple, but the model was absolutely determined to overfit. Any more complex and the training accuracy would skyrocket, while the validation data would plummet. I tried for weeks to counteract this, and even so, the model still extensively uses regularization. At the very least, the end result is extremely accurate, even if it didn't achieve and crazy accuracy in training.

In hindsight, this is probably because of how little data I was giving the network and the simplicity of the problem (a binary classification). If I combined more information, such as the depth, and asked users to assume differing postures, being able to classify things like 'slouching' or 'leaning forward' would be much more useful and impressive. I am currently working on an app to correct weightlifting form, which will be far more complex than this one.

## Where's the Data?
I don't really have a place to upload the data, and when I do something similar in the future, I'll use a different data collection system.
If you want to know how it was made: 
1. My models (friends) sat and were asked to use their computers as normal.
2. They were asked to assume either 'good' or 'bad' posture for the entire duration of the video. This allowed all the frames of an entire video to be labelled at once, rather than going through and manually marking each frame. (For our purposes, bad posture is simply any position which isn't considered traditionally good. If they are slouching, leaning, craning their neck, etc. it's considered 'bad').
3. The camera was moved continuously, changing perspective and viewing angle, to generate a variety of points.
4. Each person was asked to provide equal numbers of good and bad video. Multiple models of differing heights, body types, genders, clothing, and video background were chosen, giving the network the best chance to generalize.
5. For every frame of every video, alter it slightly (e.g. affine transform, rotation, flip).
6. Estimate the locations in 3D space using the models above, and put a label on the data.

This resulted in close to a million data points on which the neural network was trained.

## In conclusion
Overall, a very fun project that I loved creating, and my first self-directed venture into the world of neural networks. The courses provided by OpenCV were invaluable to my learning, and I'll certainly continue down the path of neural networks in the future, especially after the experience I gained as a beginner going into this project.
