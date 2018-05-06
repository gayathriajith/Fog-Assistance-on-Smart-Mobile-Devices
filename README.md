# Fog-Assistance-on-Smart-Mobile-Devices
Single image haze removal using dark channel prior. A mobile device captures the foggy image and it is defogged in a server. The client takes foggy image in an Android Application. The server side development includes image processing in Python using libraries such as OpenCV.


The main objective of this project is to develop a user friendly android application where the android user can select the foggy image which is to be defogged either by taking the picture from the mobile camera or from the gallery of the android device.The selected image is sent to the server and the corresponding haze-free image is received on the android device after performing single image defogging.

The implementation of various modules are summarized below:
The first phase in the construction of the project is to build a user interface in eclipse that can be used for the android app. First in the app, the user have to set the ip address of the server system.Image can be selected either from the gallery of the android device or by taking the picture from the android device camera. Then the image is sent to the server and the server performs the defogging operation by the following four steps and the resulting haze free image is sent back to the android device 
• Estimation of dark channels from the original image
• Estimation of Atmospheric Light using dark Channels
• Transmission Map Estimation
• Transmission Map Refinement
• Image Reconstruction
