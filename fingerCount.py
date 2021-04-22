# organize imports
import cv2
import imutils
import numpy
import math 
from sklearn.metrics import pairwise

background = None 

#vc is an object that gets input from the video camera 
vc = cv2.VideoCapture(0)

#starting on zero frames
frame = 0

#loop while the camera is open or running
while(vc.isOpened()):
 
        #The first part of the algorithm, processes and reads frame-by-frame
        #reads the image 
        (_, screen) = vc.read()
        # scales the image from the camera to 850 width without reducing pixel quality (easier to see onscreen) 
        screen = imutils.resize(screen, width=850)

        # flip the image to avoid mirror view 
        screen = cv2.flip(screen, 1)
        # Focusing on a specific portion of the screen (placement of hand into the rectangle) 
        rectangle = screen[10:450, 350:600]
        # Converts the rectangle portion (for hand) into grayscale and blurred, in order to remove any disturbances
        gray_scale = cv2.cvtColor(rectangle, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray_scale, (7, 7), 0)

        #Identify the background, by taking a frame from the first 70 frames
        if frame < 70:
            if background is None:
               background = blur.copy().astype("float")
            # Converts the grayscale image into 32 float point, in order to store the image as data
            # The data is in the RGBE Image Format 
            else:
               cv2.accumulateWeighted(blur, background, 0.5)
               # updates the background image by comparing the current frame, to the past frame
               # the third parameter, is the weighted sum of the input image meaning how fast the update speed is for the background
        else:
            # takes the current image, and compares it to the stored background image to find the difference (in byte form) between the object in front and the background 
            foreground = cv2.absdiff(background.astype("uint8"), blur)
          
            #The foreground image is a gradiant or shade of gray, and must be seperated completely into black and white
            #The threshhold value (21) is to seperate the foreground image into a binary, black and white image
            thresh_holdimg = cv2.threshold(foreground, 21, 255, cv2.THRESH_BINARY)[1]
            thresh_holdimg = cv2.dilate(thresh_holdimg, None, iterations=2)
            thresh_holdimg = cv2.erode(thresh_holdimg, None, iterations=2)

            #gets the contour of the final foreground image 
            (contours,_) = cv2.findContours(thresh_holdimg.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            #intialize contour variable of the hand 
            hand_outline = None
            
            #If statement to check for the largest contour area or object inside of the image (determining which object is the hand) 
            if len(contours)>0:
                hand_outline = max(contours, key=cv2.contourArea)

            #This If statement will only run if there is an object that enters the rectangle region 
            if hand_outline is not None:
                #Using drawContours, the outline of the hand can be displayed
                cv2.drawContours(screen, [ hand_outline + (350, 10)], -1, (95, 76, 255), 3)
                # different RGB color schemes for hand outline
                # light pink 255, 76, 95
                # dark red 36, 19, 181
              
                #Finding Convexity Defects
                con_fix = 0.0005*cv2.arcLength(hand_outline ,True)
                average = cv2.approxPolyDP(hand_outline,con_fix,True)

                #Given the contour area, use cv2.convexHull to get the convex hull 
                con_hull = cv2.convexHull(hand_outline)

                #Using the area of the contour and the area of the convex hull, calculating the ratio will allow the algorithm to identify the special case of when no fingers are held up
                #When the hand has no fingers held up (it should indicate 0) 
                a_hull = cv2.contourArea(con_hull)
                a_contour = cv2.contourArea(hand_outline)
                a_ratio=((a_hull-a_contour)/a_contour)*100

                
                con_hull = cv2.convexHull(average, returnPoints=False)
                def_area = None
                def_area = cv2.convexityDefects(average, con_hull)

                #start finger_num is 1 defect intially 
                finger_num = 1
                if (def_area.shape[0] is not None):
                #looping through each index in the def_area two dimensional array 
                  for i in range(def_area.shape[0]):  #calculating for the number of fingers
                # sp and ep is the distance from finger to finger, and fp is the fartheset point of the contour of the hand
                # d is the distance from the convex hull (between the fingers) to fp (farthest point of the contour)
                    sp, ep, fp, d = def_area[i][0]
                    start_point = tuple(average[sp][0])
                    end_point = tuple(average[ep][0])
                    far_point = tuple(average[fp][0])
                  #storing the values of sp, ep, and fp into an array where the values are unchangeable (remain constant) 

                  
                    s1 = math.sqrt((end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2)
                    s2 = math.sqrt((far_point[0] - start_point[0]) ** 2 + (far_point[1] - start_point[1]) ** 2)
                    s3 = math.sqrt((end_point[0] - far_point[0]) ** 2 + (end_point[1] - far_point[1]) ** 2)
                    angle_defect = math.acos((s2 ** 2 + s3 ** 2 - s1 ** 2) / (2 * s2 * s3))
                    if (angle_defect <= math.pi / 2):  # Counting fingers whose angles are under 90 in order to look at the defects which are between the fingers
                     finger_num += 1
                  if finger_num==1:
                        if a_ratio<16:
                           finger_num = 0
                        else:
                           finger_num = 1
                cv2.putText(screen, 'Finger Counter', (70,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2) 
                cv2.putText(screen, str(finger_num), (70,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2) 
                
                #final foreground image 
                cv2.imshow("Final Foreground Image", thresh_holdimg)

        #show rectangle portion 
        cv2.rectangle(screen, (600, 10), (350, 450), (73,255,248), 2)

        #incrementing the frames
        frame += 1.5

        #displaying the video camera
        cv2.imshow("Video Feed", screen)

        #using window key to stop while loop
        if cv2.waitKey(1) & 0xFF == ord("e"):
            break

# release video
vc.release()
cv2.destroyAllWindows()
