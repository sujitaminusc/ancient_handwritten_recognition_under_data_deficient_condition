from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import load_model

#test data rescaled.
# since we only working on non-rgb images.
test_input = ImageDataGenerator(rescale= 1/255.0)

train_img_path = "Sanskrit/sanskrit_test_data"
num_classes = 58
batch_size = 8
train_gen = test_input.flow_from_directory(
    directory=train_img_path,
    target_size=(28, 28),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    subset='training',
    shuffle=True,
    seed=42
)

model_path = 'sanskrit_pickle.hdf5'

print("Please wait.....")
best_model = load_model(model_path)

best_model.load_weights(model_path)
print("Loaded model from disk")
score = best_model.evaluate(train_gen, verbose=0)
print('Test accuracy percentage: ' + str(score[1] * 100) + "%")