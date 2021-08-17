import tensorflow as tf
import tensorflow_hub as hub 
import streamlit as st

class Model:
    def load_model(self, model_path):
        """
        Loads a saved model from specified path
        """

        print(f"Loading saved model from: {model_path}...")
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={"KerasLayer": hub.KerasLayer}
        )

        return model
    
    def process_image(self, image_path, img_size=24):
        """
        Takes an image file path and turns the image into a Tensor
        """

        #read an image file into a variable
        image = tf.io.read_file(image_path)

        #turn image into tensors
        image = tf.image.decode_jpeg(image, channels=3) #3 for Red Green Blue

        #convert color channel values from 0 - 255 to 0 - 1 values (decimals)
        image = tf.image.convert_image_dtype(image, tf.float32)

        #resize image to (224, 224)
        image = tf.image.resize(image, size=[img_size, img_size])

        #return the image
        return image
    def get_image_label(self, image_path, label):
        """
        Takes an image file path name and the associated label, 
        and processes the image and returns a tuple of (image, label)
        """
        image = process_image(image_path)

        return image, label

    def create_data_batches(self, X, y=None, batch_size=32, valid_data=False, test_data=False):
        """
        Create batches of data out of image (X) and label(y) pairs.
        Shuffles the data if it's training data but doesn't shuffle if it's validation data.
        Also accepts test data as input (no labels)
        """

        if test_data:
            print("Creating test data batches...")
            data = tf.data.Dataset.from_tensor_slices((tf.constant(X)))
            data_batch = data.map(process_image).batch(batch_size)

            return data_batch
        elif valid_data:
            #if data is a valid data set, no need to shuffle
            print("Creating validation data batches...")
            data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
            data_batch = data.map(get_image_label).batch(batch_size)

            return data_batch
        else:
            print("Creating training data batches...")
            #turn filepaths and labels into tensors
            data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
            #shuffle pathnames and labels
            data = data.shuffle(buffer_size=len(X))
            #create (image, label) tuples, also turns image path into a preprocessed image
            data = data.map(get_image_label)

            data_batch = data.batch(batch_size)
            return data_batch

model = Model().load_model('model\20210704-01-M1625361368-full-image-set-mobilenetv2-Adam.h5')
file = st.file_uploader("Select an image to classify")
data = Model().create_data_batches(test_data=True, X=file)
