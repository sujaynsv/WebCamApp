# Webcam Capture & Analysis Application

## Description
This is a **Webcam Capture & Analysis Application** built using **OpenCV** and **Streamlit**. The application enables users to capture images from their webcam and apply real-time filters such as:

- **Edge Detection**: Detects edges in the frame using the Canny algorithm.
- **Face Detection**: Identifies and highlights faces using OpenCV's Haar cascade classifier.
- **Motion Detection**: Detects movement by comparing frame differences.

The application is designed to be **interactive and user-friendly**, allowing users to choose the desired feature from a sidebar menu and capture images with just one click.

## Features
- **Live webcam feed display**
- **Real-time image processing filters**
- **User-friendly Streamlit interface**
- **One-click image capture & save functionality**

## Installation
### Prerequisites
Ensure you have Python installed (preferably **Python 3.8+**).

### Install Required Packages
Run the following command to install the dependencies:
```bash
pip install opencv-python numpy streamlit
```

## Usage
1. Run the application using Streamlit:
   ```bash
   streamlit run app.py
   ```
2. Allow webcam access when prompted.
3. Choose a feature from the sidebar (None, Edge Detection, Face Detection, or Motion Detection).
4. Click **"Capture Image"** to save the processed image.

## File Structure
```
|-- app.py                # Main application script
|-- captured_images/      # Directory where captured images are saved
|-- requirements.txt      # List of dependencies (optional)
```

## Future Enhancements
- **Hand Gesture Recognition** for interactive control.
- **Object Detection** using deep learning models.
- **Background Removal** for virtual backgrounds.

---
Developed with ❤️ using OpenCV & Streamlit.

