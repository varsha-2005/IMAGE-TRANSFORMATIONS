# IMAGE-TRANSFORMATIONS

## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:

Import the necessary libraries and read the original image and save it as a image variable.

### Step2:

Translate the image using a function warpPerpective()

### Step3:

Scale the image by multiplying the rows and columns with a float value.

### Step4:

Shear the image in both the rows and columns.

### Step5:

Find the reflection of the image.

### Step6:

Rotate the image using angle function.

## Program:

#### Developed By:Varsha.G
#### Register Number: 212222230166

## i)Image Translation

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt

input_image=cv2.imread("vijay.png")
input_image=cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(input_image)
plt.show()
rows,cols,dim=input_image.shape
M=np.float32([[1,0,50],  [0,1,100],  [0,0,1]])
translated_image=cv2.warpPerspective(input_image,M,(cols,rows))
plt.axis('off')
plt.imshow(translated_image)
plt.show()
```

## ii) Image Scaling

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
org_image = cv2.imread("vijay.png")
org_image = cv2.cvtColor(org_image,cv2.COLOR_BGR2RGB)
plt.imshow(org_image)
plt.show()
rows,cols,dim = org_image.shape
M = np.float32([[1.5,0,0],[0,1.7,0],[0,0,1]])
scaled_img = cv2.warpPerspective(org_image,M,(cols*2,rows*2))
plt.imshow(org_image)
plt.show()
```


## iii)Image shearing

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
org_image = cv2.imread("vijay.png")
org_image = cv2.cvtColor(org_image,cv2.COLOR_BGR2RGB)
plt.imshow(org_image)
plt.show()
rows,cols,dim = org_image.shape
M_X = np.float32([[1,0.5,0],[0,1,0],[0,0,1]])
M_Y = np.float32([[1,0,0],[0.5,1,0],[0,0,1]])
sheared_img_xaxis = cv2.warpPerspective(org_image,M_X,(int(cols*1.5),int(rows*1.5)))
sheared_img_yaxis = cv2.warpPerspective(org_image,M_Y,(int(cols*1.5),int(rows*1.5)))
plt.imshow(sheared_img_xaxis)
plt.show()
plt.imshow(sheared_img_yaxis)
plt.show()
```


### iv)Image Reflection

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
org_image = cv2.imread("vijay.png")
org_image = cv2.cvtColor(org_image,cv2.COLOR_BGR2RGB)
plt.imshow(org_image)
plt.show()
rows,cols,dim = org_image.shape
M_X = np.float32([[1,0,0],[0,-1,rows],[0,0,1]])
M_Y = np.float32([[-1,0,cols],[0,1,0],[0,0,1]])
reflected_img_xaxis = cv2.warpPerspective(org_image,M_X,(int(cols),int(rows)))
reflected_img_yaxis = cv2.warpPerspective(org_image,M_Y,(int(cols),int(rows)))
plt.imshow(reflected_img_xaxis)
plt.show()
plt.imshow(reflected_img_yaxis)
plt.show()
```


### v)Image Rotation
```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
input_image = cv2.imread("vijay.png")
input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(input_image)
plt.show()

angle=np.radians(10)
M=np.float32([[np.cos(angle),-(np.sin(angle)),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
rotated_img = cv2.warpPerspective(input_image,M,(int(cols),int(rows)))

plt.imshow(rotated_img)
plt.show()
```



### vi)Image Cropping
```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
org_image = cv2.imread("vijay.png")
org_image = cv2.cvtColor(org_image,cv2.COLOR_BGR2RGB)
plt.imshow(org_image)
plt.show()
rows,cols,dim = org_image.shape
cropped_img=org_image[80:900,80:500]
plt.imshow(cropped_img)
plt.show()
```





## Output:
### i)Image Translation

![image](https://github.com/TejaswiniGugananthan/IMAGE-TRANSFORMATIONS/assets/121222763/c6080119-a657-4e90-887c-96d5cab5fc44)


### ii) Image Scaling

![image](https://github.com/TejaswiniGugananthan/IMAGE-TRANSFORMATIONS/assets/121222763/442f2b3a-44d1-4a99-b6ee-9bb22ef875a7)


### iii)Image shearing

![image](https://github.com/TejaswiniGugananthan/IMAGE-TRANSFORMATIONS/assets/121222763/9c728086-ad6c-4451-90fe-0924af49768f)

![image](https://github.com/TejaswiniGugananthan/IMAGE-TRANSFORMATIONS/assets/121222763/ce3b580f-ec06-4850-96d2-44eb1d648ed4)


### iv)Image Reflection

![image](https://github.com/TejaswiniGugananthan/IMAGE-TRANSFORMATIONS/assets/121222763/1257a902-3eca-45df-93c6-7e29177377f0)

![image](https://github.com/TejaswiniGugananthan/IMAGE-TRANSFORMATIONS/assets/121222763/fbc9b170-a9ae-46ce-b8a4-99ac080a5d97)



### v)Image Rotation

![image](https://github.com/TejaswiniGugananthan/IMAGE-TRANSFORMATIONS/assets/121222763/6579ffc5-9b22-4d84-984b-2464a5b69982)



### vi)Image Cropping

![image](https://github.com/TejaswiniGugananthan/IMAGE-TRANSFORMATIONS/assets/121222763/8583fc8c-d956-4691-959b-7e60c8403176)




## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
