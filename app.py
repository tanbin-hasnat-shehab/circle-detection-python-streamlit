import streamlit as st
import cv2 as cv
from PIL import Image,ImageEnhance
import numpy as np
import os
import PIL
import numpy


def detect(image,x_r,sh,con):
    '''
    Function to detect circles
    '''
    pil_image = image.convert('RGB') 
    img = numpy.array(pil_image) 
    # Convert RGB to BGR 
    img = img[:, :, ::-1].copy()
    
    output = img.copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)
    


    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 0.5, 1,
                              param1=50, param2=30, minRadius=10, maxRadius=20)
    detected_circles = np.uint16(np.around(circles))
    for (x, y ,r) in detected_circles[0, :]:
        cv.circle(output, (x, y), r+5, (0, 0, 0), -1)
        #cv.circle(output, (x, y), 2, (0, 255, 255), 3)
    cv.imwrite('0.png',output)
    
        
    img=cv.imread('0.png')
    output = img.copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 9)

    
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 0.5, 1,
                                  param1=50, param2=30, minRadius=min_rad, maxRadius=20)
    detected_circles = np.uint16(np.around(circles))
    for (x, y ,r) in detected_circles[0, :]:
        cv.circle(output, (x, y), r+5, (0, 0, 0), -1)
        #cv.circle(output, (x, y), 2, (0, 255, 255), 3)
    cv.imwrite('1.png',output)


    


    img=cv.imread('1.png')
    img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    _,result=cv.threshold(img,100,255,cv.THRESH_BINARY)
    adaptive=cv.adaptiveThreshold(img,x_r,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,sh,con)
    cv.imwrite('2.png',adaptive)
    #os.startfile('2.png')


  
    
    return adaptive
    
    



def about():
    st.write(
        '''
        contact email - tanbinhasnat04@gmail.com
        ''')


def main():
    st.title("circle Detection App :sunglasses: ")
    st.write("**Using the Haar cascade Classifiers**")

    activities = ["Home", "About"]
    choice = st.sidebar.selectbox("Pick something fun", activities)

    if choice == "Home":

        st.write("Go to the About section from the sidebar to learn more about it.")
        
        # You can specify more file types below if you want
        image_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'])
        print(type(image_file))
        global min_rad
        min_rad=st.slider('mir radius : ',value=5)
        st.write('min radius ',min_rad)

        br=st.slider('Brightness : ',value=1.0,min_value=1.0,max_value=4.0,step=0.1)
        st.write('Brightness ',br)


        sh=st.slider('parameter 1 : ',value=111,min_value=50,max_value=200,step=10)
        con=st.slider('parameter 2 : ',value=20,min_value=0,max_value=50,step=5)
        

        
        

        x_r=st.slider('Black and white : ',value=250,min_value=200,max_value=255,step=5)
        




        if image_file is not None:

            image = Image.open(image_file)

            
            enhencer=ImageEnhance.Brightness(image)
            image=enhencer.enhance(br)
            

            


            if st.button("Process"):

                
                # result_img is the image with rectangle drawn on it (in case there are faces detected)
                # result_faces is the array with co-ordinates of bounding box(es)
                result_img = detect(image=image,x_r=x_r,sh=sh,con=con)
                st.image(result_img, use_column_width = True)

    elif choice == "About":
        about()




if __name__ == "__main__":
    main()
