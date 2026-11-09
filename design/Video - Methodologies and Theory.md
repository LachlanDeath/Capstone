# Methodologies, Theory, and Design Decisions
### High Level
* Define a "correct" range of posture for the cello player - This posture is a range of playing positions in the shoulder, arm, eblow, wrist, and hand, that we define as "correct" paying posture. 
* Using a machine learning computer vision model, we can identify, track, and "compute" this posture.
* A computer vision model is trained on human movement and images, such that it can easily identify these waypoints.
* The Google Mediapipe model is an open-source model that is already trained. It can easily be implemented by downloading and appropriately setting up an appropiate workspace on your own computer.
* Building a model that can track the cello, with refernce to the player, and 
* We  angles between waypoints on the arm (shoulder, elbow, wrist, hand)

**Defining "Correct" posture.** 
  * How our team mathematically and fundmentally defines posture is key to this project. We have to account for egonomics, and ensuiring the player maintains in a healthy, productive playing position. All of this (healthy, productive, etc.) needs to be theoretiucally informed        
 
 **Model Performance - "Beneath" the surface**
* Defining metric for successful model performance. Ensuring good practices for data annotation, and storage (Keeping good ethics). 
* This will include computations for the processing and training loads. Do we require additional hardware? Specific softwares? 
* Whole program should be packaged nicely, and function effectively.                                                                                             
**Presentation and User Iterface/Experience**
* As this is a user-centred project, the presentation to the user needs to follow research-grounded design
* Optimizing users experience for maximum effectively learning using the tool, while 
* How do we present corrections effectively?

### Google Media Pipe

### YOLO Model