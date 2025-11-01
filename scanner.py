"""
Classical document scanner using OpenCV — detects quadrilateral objects and applies perspective correction.
No machine learning used. Academic lab project.
"""

import cv2
import numpy as np

# =============================================================================
# Function Definitions
# =============================================================================

def cornerDraw(img, contours, isSorted):
    '''
    This is a helper function to draw the corner points on the image.

    Parameters
    ----------
    img : cv2.image
        The image where the points will be drawn.
    contours : array-like of shape (4, 2)
        Contains the positional values of the four corners.
    isSorted : Boolean
        Boolean for checking whether the corner positions are sorted or not.

    Returns
    -------
    None.

    '''
    imgCopy = img.copy()
    
    # Labels are used to write on the screen
    labels = ['P0', 'P1', 'P2', 'P3']
    
    for i, (x, y) in enumerate(contours):
        center = (int(x), int(y))
        cv2.circle(imgCopy, center, 8, (255 ,0 ,0), -1)
        cv2.putText(imgCopy, f'{labels[i]} {int(x), int(y)}', (int(x) + 10, int(y) - 10), 0, 0.5, (0,0,255), 2)
    
    # Image display; captions vary based on whether corners are sorted or not.
    if isSorted == 0: cv2.imshow("Corner points unsorted", imgCopy)
    elif isSorted == 1: cv2.imshow("Corner points sorted", imgCopy)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return None


def findEdge(img):
    '''
    Detects the edges over the whole image.

    Parameters
    ----------
    img : cv2.image
        The image where the edge operations will be performed.

    Returns
    -------
    edgeClosed : cv2.image
        The binary image with the edges dilated and closed.

    '''
    
    # Edge is detected using Canny; threshold: 50 -150
    edgeFind = cv2.Canny(img, 50, 150)
    
    # Edges are dilated to connect the broken edges
    edgeDilate = cv2.dilate(edgeFind, np.ones((3,3), np.uint8), iterations=1)
    
    # Edges are closed to fill any small gaps
    edgeClosed = cv2.morphologyEx(edgeDilate, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    
    cv2.imshow('Canny', edgeFind)
    cv2.imshow('Edges', edgeDilate)
    cv2.imshow('Thick', edgeClosed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return edgeClosed


def findContourApproxi(contours):
    '''
    Returns the contour with the approximate shape of a four-sided polygon.

    Parameters
    ----------
    contours : list
        The list of contours of the whole image.

    Returns
    -------
    approx : np.ndarray or None
        Approximated contour as a NumPy array of shape (4, 1, 2) if a quadrilateral is found, otherwise None.

    '''
    for contour in contours:
        for factor in range(1, 11):
            factor /= 100
            epsilon = factor * cv2.arcLength(contour,True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                return approx

    return None


def findBoundary(img):
    '''
    Finds the boundary of the document using contours.

    Parameters
    ----------
    img : cv2.image
        The binray image used to detect the boundary.

    Returns
    -------
    contourFix : list
        The contour with the specified four corners.

    '''
    
    # Contours detected and sorted to find the largest contour first
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Drawing contours
    drawImg = np.zeros_like(img)
    cv2.drawContours(drawImg, contours, -1, 255, 1)
    cv2.imshow("Contoured section", drawImg)

    # The largest contour with the four corners, approximating a document is determined
    contourFix = findContourApproxi(contours)
    if contourFix is None:
        raise ValueError("No quadrilateral contour found in the image.")
        
    # Drawing the largest contour, approximated into four sided polygon
    singleBoundary = np.zeros_like(img)
    cv2.drawContours(singleBoundary, [contourFix], -1, 255, 3)
    cv2.imshow("Contour fixed section", singleBoundary)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return contourFix

    
def contourPointsOrder(contPts):
    '''
    Sorts the contours for the transformation operation.

    Parameters
    ----------
    contPts : np.array(4, 2)
        The corners points to be sorted.

    Returns
    -------
    rect : np.array(4, 2)
        Returns the sorted corners.

    '''
    rect = np.zeros((4, 2), dtype="float32")
    contSum = contPts.sum(axis=1)
    print(f'\nsum: {contSum}')
    
    # Finding the top-left and bottom-right corner
    rect[0] = contPts[np.argmin(contSum)]
    rect[2] = contPts[np.argmax(contSum)]
    
    diff = np.diff(contPts, axis=1)
    print(f'\ndiff: {diff}')
    
    # Finding the top-right and bottom-left corner
    rect[1] = contPts[np.argmin(diff)]
    rect[3] = contPts[np.argmax(diff)]
    
    return rect


def findMaxSize(points):
    '''
    Determines the size of the transformed document.

    Parameters
    ----------
    points : np.array(4, 2)
        The four sorted corners used to determine height and width.

    Returns
    -------
    maxWidth : float
        The maximum width of the document.
    maxHeight : float
        The maximum height of the document.

    '''
    # Finding the top and bottom widths
    (topLeft, topRight, bottomRight, bottomLeft) = points
    widthA = np.sqrt(((bottomRight[0] - bottomLeft[0]) ** 2) + ((bottomRight[1] - bottomLeft[1]) ** 2))
    widthB = np.sqrt(((topRight[0] - topLeft[0]) ** 2) + ((topRight[1] - topLeft[1]) ** 2))
    
    # Finding the left and right heights
    heightA = np.sqrt(((topRight[0] - bottomRight[0]) ** 2) + ((topRight[1] - bottomRight[1]) ** 2))
    heightB = np.sqrt(((topLeft[0] - bottomLeft[0]) ** 2) + ((topLeft[1] - bottomLeft[1]) ** 2))

    # Finding the maximum values
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    
    return maxWidth, maxHeight

    
# =============================================================================
# The actual operations
# =============================================================================

# Step1: Reading and resizing the image
img = cv2.imread('test (3).jpeg')

if img is None:
    raise ValueError("Image not found. Check the file path and name.")

# Scaling down is used for visualizing only

# scaleDown = 0.60  
# resizedImg = cv2.resize(img, None, fx = scaleDown, fy = scaleDown, interpolation = cv2.INTER_LINEAR)

resizedImg = img.copy()
orig = resizedImg.copy()

# Step2: Grayscaling and blurring
grayImg = cv2.cvtColor(resizedImg, cv2.COLOR_BGR2GRAY)
blurredImg = cv2.GaussianBlur(grayImg, (5, 5), 0)

cv2.imshow('Original Image', orig)
cv2.imshow('Blurred', blurredImg)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step3: Detecting edges with Canny
edgeDilated = findEdge(blurredImg)

# Step4: Detecting the boundary using contour
boundaryContour = findBoundary(edgeDilated)
cv2.drawContours(resizedImg, [boundaryContour], -1, (0, 255, 0), 3)
cv2.imshow("Object boundary", resizedImg)

# Step5: Finding and sorting the corner points
contourArray = np.reshape(boundaryContour, (4, 2))
cornerDraw(resizedImg, contourArray, 0)

sortedCorners = contourPointsOrder(contourArray)
cornerDraw(resizedImg, sortedCorners, 1)

# Step6: Finding height and width
height, width = findMaxSize(sortedCorners)

# Step7: Finding transformed image size
transformed = np.array([
    [0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]
    ], dtype="float32")

# Step8: Performing transformation
pers = cv2.getPerspectiveTransform(sortedCorners, transformed)
warped = cv2.warpPerspective(orig, pers, (width, height))

cv2.imshow("Warped image", warped)

cv2.waitKey(0)
cv2.destroyAllWindows()