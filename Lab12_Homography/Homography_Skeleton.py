import numpy as np
import cv2
import matplotlib.pyplot as plt

FIGURE_SIZE = (12, 9)
imgFolder = ''
imgName = 'chessboard_perspective.jpg'

img_persp = plt.imread(imgFolder + imgName)

# Coordinates are defined as [vertical, horizontal]
if imgName == "chessboard_perspective.jpg":
    Ps = np.array([[45, 385], [420, 55], [665, 723], [187, 928]])   # TODO: Additional Points go here
    points_u = np.array([[0, 0], [700, 0], [700, 700], [0, 700]])   # TODO: Coordinates of undistorted Points go here

def plot_points_on_image(img: np.ndarray, points: np.ndarray):
    plt.figure()
    plt.imshow(img)
    for point in points:
        plt.plot(point[1], point[0], 'o')    
    plt.title("Image with Points at Specified Coordinates")    
    plt.xlabel("Coordinate 1")
    plt.ylabel("Coordinate 0")
    plt.show()

plot_points_on_image(img_persp, Ps)

# Create the matrix A
A = np.zeros((8, 8))
y = np.zeros((8,))

for i in range(len(Ps)):
    y_d, x_d = Ps[i]
    y_u, x_u = points_u[i]
    y[2 * i] = x_d
    y[2 * i + 1] = y_d
    A[2 * i, :] = [x_u, y_u, 1, 0, 0, 0, -x_u*x_d, -y_u*x_d]
    A[2 * i + 1, :] = [0, 0, 0, x_u, y_u, 1, -y_d*x_u, -y_d*y_u]

# Solve the system of linear equations
solution = np.linalg.solve(A,y)
H = np.array([[solution[0], solution[1], solution[2]],
              [solution[3], solution[4], solution[5]],
              [solution[6], solution[7], 1]])

H = np.linalg.inv(H)
# Undistort the image
undistorted_img = cv2.warpPerspective(img_persp, H, (img_persp.shape[0], img_persp.shape[0]))

# Display the undistorted image using cv2.imshow
cv2.imshow("Undistorted Image", undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()