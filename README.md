# Image-Captioning-Visually-Impaired
This code demonstrates an image captioning application that uses a camera feed to capture an image and generate a caption describing the content of the image. The application is built using the Streamlit framework, the Hugging Face Transformers library for pre-trained models, and OpenCV for image processing.

Instructions
Follow these steps to run the image captioning application:

1.Install Dependencies: Before running the code, make sure you have the required libraries installed. You can install them using the following command:


"pip install streamlit opencv-python-headless transformers torch pillow"

2.Clone or Download the Code: Clone this repository or download the code as a ZIP file and extract it to a local directory.

3.Navigate to the Directory: Open a terminal and navigate to the directory where you have the code files.

4.Run the Application: In the terminal, run the following command:

"streamlit run app.py"

app.py with the actual name of the file containing the provided code

5.Camera Access: The application will attempt to access your camera. A window displaying the camera feed will appear. If the camera is not accessible, an error message will be displayed.

6.Capture and Predict: Click the "Capture and Predict" button to capture an image from the camera feed. The captured image will be displayed along with the predicted caption describing the content of the image.

7.Generate Caption: The application uses a pre-trained image captioning model to generate captions. The caption generation process might take a moment.

8.Exit the Application: You can close the application's window to stop the camera feed and exit the program.
