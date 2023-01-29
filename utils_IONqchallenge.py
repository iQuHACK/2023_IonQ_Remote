import numpy as np
import matplotlib.pyplot as plt
from resizeimage import resizeimage

# function 0.
def plot_image(img, title: str):
    
    """ Function for plotting the image using matplotlib."""
    
    plt.title(title)
    plt.xticks(range(img.shape[0]))
    plt.yticks(range(img.shape[1]))
    plt.imshow(img, extent=[0, img.shape[0], img.shape[1], 0], cmap='viridis')
    plt.show()

# function 1.
def image_normalization( dataset , size , show ):
    
    """ this function brings values of uimage pixels betoween 0 and 1 in a flatten numpy array format."""   
    
    for ele in dataset:
        ele /= np.max(ele)
          
    return dataset.reshape( len(dataset) , size ** 2 )


# function 3.
def get_count_of_pixel( arr1 , arr2 ):
    
    same = 0
    notsame = 0
    
    for i in range( len( arr1 ) ):
        if arr1[ i ] == arr2[ i ]:
            same += 1
        else:
            notsame += 1
            
    return ( same , notsame )

# function 4.
def amplitude_encode(img_data):
    
    # Calculate the RMS value
    rms = np.sqrt( np.sum( np.sum( img_data**2 , axis = 1 ) ) )
    
    # Create normalized image
    image_norm = []
    for arr in img_data:
        for ele in arr:
            image_norm.append( ele / rms )
    
    # Return the normalized image as a numpy array
    return np.array(image_norm)