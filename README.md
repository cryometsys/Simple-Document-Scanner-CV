# Simple Document Scanner Using Classical Computer Vision

A non-machine learning-based document scanner that detects a rectangular object (e.g., paper, ID card, or sheet) in a photograph and transforms it into a clean, front-facing, scanned-like image using only classical image processing techniques.

> **Academic Lab Project** – Built under constraints prohibiting the use of machine learning.

## Objective
- Automatically detect the largest quadrilateral (assumed to be a document) in an image.
- Extract and correctly order its four corners.
- Apply perspective correction to simulate a flatbed scanner output.

## Tools & Libraries
- **Python 3.x**
- **OpenCV** (`cv2`) – for image processing, edge detection, contour analysis, and perspective warping
- **NumPy** – for numerical operations and array handling

## Project Structure
```
document-scanner/
├── scanner.py                # Main implementation script
├── test (4).jpeg             # Sample input image
├── README.md
```

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/cryometsys/Simple-Document-Scanner-CV.git
   cd Simple-Document-Scanner-CV
   ```
2. Install dependencies:
   ```bash
   pip install opencv-python numpy
   ```
3. Run the script:
   ```bash
   python scanner.py
   ```
> Make sure the input image is in the same directory, or update the filename in the code.

## Methodology Overview
The pipeline consists of five key classical CV steps:

1. **Preprocessing**: Convert to grayscale and apply Gaussian blur to reduce noise.
2. **Edge Detection**: Use Canny edge detector followed by morphological closing to form continuous boundaries.
3. **Contour Approximation**: Find the largest contour and approximate it to a 4-point polygon.
4. **Corner Sorting**: Order corners as [top-left, top-right, bottom-right, bottom-left] using coordinate sums/differences.
5. **Perspective Warping**: Apply a homography transform to produce a flat, undistorted output.

> No neural networks, no training data—just geometry, filtering, and OpenCV.

## Notes
- Works best when the document is the **dominant object** with **clear contrast** against the background.
- Assumes the target is **approximately rectangular** and **not heavily occluded**.
- Designed for educational purposes to showcase classical computer vision techniques.

## References
- OpenCV Documentation: [https://docs.opencv.org](https://docs.opencv.org)
- Canny Edge Detection (1986)
- Perspective Transform in OpenCV

## Academic Context
Developed as part of CSE 4128: Image Processing and Computer Vision Laboratory

Khulna University of Engineering & Technology
