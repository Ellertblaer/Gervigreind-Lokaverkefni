# Importar gognunum
! wget https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/fk6rys63h9-1.zip
! unzip fk6rys63h9-1.zip
! unzip Test_images.zip
! unzip Training_images.zip
! unzip Validation_images.zip

# InceptionV3, stór hluti kemur frá Google AI Studio
IMG_HEIGHT = 299, IMG_WIDTH = 299
BATCH_SIZE = 32
EPOCHS = 25 
LEARNING_RATE = 0.0001

try:
    train_dataset = image_dataset_from_directory(
        train_dir,
        label_mode='binary',
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42 )
        
    validation_dataset = image_dataset_from_directory(
        validation_dir,
        label_mode='binary',
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=False )

    test_dataset = image_dataset_from_directory(
        test_dir,
        label_mode='binary',
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=False )

    class_names = train_dataset.class_names
    print("Class Names:", class_names)

    for image_batch, labels_batch in train_dataset.take(1):
        print("Image batch shape:", image_batch.shape)
        print("Label batch shape:", labels_batch.shape)
        break

except FileNotFoundError:
    print("\nError: One of the data directories (Training, Validation, Testing) was not found.")
    print("Please ensure the dataset was extracted correctly and the base_dir is set properly.")
    



#----------------------------------------------------
# Configure Dataset for Performance & Preprocessing
#----------------------------------------------------
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

preprocess_input = tf.keras.applications.inception_v3.preprocess_input



#----------------------------------------------------
# Build the Model
#----------------------------------------------------

# Loada InceptionV3 módelið
base_model = InceptionV3(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                         include_top=False, # Exclude the final classification layer
                         weights='imagenet')

base_model.trainable = False

inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = preprocess_input(inputs) 
x = base_model(x, training=False) # Grunnmódelið
x = GlobalAveragePooling2D()(x)  # Bæta ofan á InceptionV3 hlutann
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x) 
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)

#----------------------------------------------------
# Compile the Model
#----------------------------------------------------
model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]) 
print("Model compiled.")

#----------------------------------------------------
# Train the Model
#----------------------------------------------------

early_stopping = EarlyStopping(monitor='val_loss', patience=5, # Hætta ef val_loss batnar ekki eftir 5 epochs
                               restore_best_weights=True)
model_checkpoint = ModelCheckpoint(filepath='best_cad_model.keras', # Vista besta módelið
                                   save_best_only=True,
                                   monitor='val_loss',
                                   mode='min')
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset,
    callbacks=[early_stopping, model_checkpoint]
)
