import streamlit as st
import cv2 as cv
from PIL import Image,ImageEnhance
import numpy as np
import os
import PIL
import numpy


def detect(image):
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


    


    


  
    
    return output
    
    



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

        br=st.slider('Brightness : ',value=1.4,min_value=1.0,max_value=4.0,step=0.1)
        st.write('Brightness ',br)

        con=st.slider('contrast : ',value=1.5,min_value=1.0,max_value=4.0,step=0.1)
        st.write('contrast ',con)

        sh=st.slider('sharpness : ',value=1.0,min_value=1.0,max_value=4.0,step=0.1)
        st.write('sharpness ',sh)


        if image_file is not None:

            image = Image.open(image_file)

            
            enhencer=ImageEnhance.Brightness(image)
            image=enhencer.enhance(br)
            enhencer=ImageEnhance.Contrast(image)
            image=enhencer.enhance(con)
            enhencer=ImageEnhance.Sharpness(image)
            image=enhencer.enhance(sh)

            


            if st.button("Process"):

                
                # result_img is the image with rectangle drawn on it (in case there are faces detected)
                # result_faces is the array with co-ordinates of bounding box(es)
                result_img = detect(image=image)
                st.image(result_img, use_column_width = True)

    elif choice == "About":
        about()




if __name__ == "__main__":
    main()