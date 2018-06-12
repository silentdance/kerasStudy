from keras.models import load_model
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

model = load_model("model.h5")

def predictImage(path):
    img = load_img(path)
    img = img.resize((150,150),Image.ANTIALIAS)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    return model.predict(x)


print('c1.jpg:',predictImage('../cat and dog/c1.jpg'))
print('c2.jpg:',predictImage('../cat and dog/c2.jpg'))
print('c3.jpg:',predictImage('../cat and dog/c3.jpg'))
print('c4.jpg:',predictImage('../cat and dog/c4.jpg'))
print('c5.jpg:',predictImage('../cat and dog/c5.jpg'))
print('c6.jpg:',predictImage('../cat and dog/c6.jpg'))
print('c7.jpg:',predictImage('../cat and dog/c7.jpg'))
print('c8.jpg:',predictImage('../cat and dog/c8.jpg'))

print('================================')
print('d1.jpg:',predictImage('../cat and dog/d1.jpg'))
print('d2.jpg:',predictImage('../cat and dog/d2.jpg'))
print('d3.jpg:',predictImage('../cat and dog/d3.jpg'))
print('d4.jpg:',predictImage('../cat and dog/d4.jpg'))
print('d5.jpg:',predictImage('../cat and dog/d5.jpg'))
print('d6.jpg:',predictImage('../cat and dog/d6.jpg'))
print('d7.jpg:',predictImage('../cat and dog/d7.jpg'))
print('d8.jpg:',predictImage('../cat and dog/d8.jpg'))