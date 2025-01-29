# Face Recognition System

## Overview
This project is a real-time face recognition system using OpenCV, face_recognition, and pickle for data persistence. The system detects faces from a webcam, identifies known faces, and allows users to add, update, or delete identities interactively.

## Features
- Detects and recognizes faces in real-time from a webcam.
- Stores known face encodings and names in a `face_data.pkl` file.
- Allows adding new faces with user-defined names.
- Provides an option to update or delete existing faces.
- Displays face recognition accuracy based on the distance metric.

## Installation
### Prerequisites
Ensure you have Python installed (preferably 3.7 or later). Then, install the required dependencies:

```bash
pip install opencv-python numpy face-recognition pickle-mixin
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/Ayush0091/Face-recognition-
   cd Face-recognition-
   ```
2. Run the script:
   ```bash
   python face.py
   ```
3. The webcam will start, and faces will be detected.
4. If an unknown face appears, you'll be prompted to add it to the database.
5. If a known face appears, you can choose to update or delete it.
6. Press `q` to exit.

## How It Works
- The script loads known faces and names from `face_data.pkl`.
- Captures real-time video from the webcam.
- Detects faces and computes their encodings.
- Compares detected faces with stored encodings.
- Displays the face name and recognition accuracy.
- Prompts the user for actions if an unknown or recognized face appears.

## File Structure
```
Face-recognition-/
│── face.py             # Main script
│── face_data.pkl       # Stored face encodings and names
│── README.md           # Project documentation
```

## Dependencies
- OpenCV (`cv2`)
- NumPy (`numpy`)
- face_recognition (`face_recognition`)
- Pickle (`pickle` for saving and loading face data)

## Notes
- Ensure your webcam is functional before running the script.
- `face_data.pkl` is automatically updated when adding, updating, or deleting a face.
- The recognition threshold is set to `0.6` for balancing accuracy and flexibility.

## Future Improvements
- Support for multiple face detection at once.
- Integration with a database for cloud storage.
- Improved UI for user interaction.

## License
This project is open-source and available under the MIT License.

## Author
[Ayush0091](https://github.com/Ayush0091)  


