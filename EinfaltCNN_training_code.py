# Importar gognunum
! wget https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/fk6rys63h9-1.zip
! unzip fk6rys63h9-1.zip
! unzip Test_images.zip
! unzip Training_images.zip
! unzip Validation_images.zip

#Einfalt CNN fra grunni
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
      '/content/Training_images',
      target_size=(img_height, img_width),
      batch_size=32,
      class_mode='binary'
)

validation_generator = val_datagen.flow_from_directory(
      '/content/Validation_images',
      target_size=(img_height, img_width),
      batch_size=32,
      class_mode='binary'
)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)
