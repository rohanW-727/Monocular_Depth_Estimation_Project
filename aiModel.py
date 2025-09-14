import os
import cv2
import random as rn
import numpy as np
import tensorflow as tf
import pillow_heif
from PIL import Image
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

SEED = 10
np.random.seed(SEED); rn.seed(SEED); tf.random.set_seed(SEED)

pillow_heif.register_heif_opener()

IMG_SIZE = (256, 256)
MASK_KERNEL = np.ones((3, 3), np.uint8)  # Smaller kernel for thin lines
_model = None  # Cache the model

def load_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)

def augment_image(img: np.ndarray) -> np.ndarray:
    """Apply random augmentations to the image."""
    alpha = rn.uniform(0.9, 1.1)  # Contrast
    beta = rn.uniform(-10, 10)   # Brightness
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    angle = rn.uniform(-5, 5)    # Small rotation
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h))
    if rn.random() > 0.5:
        img = cv2.flip(img, 1)  # Horizontal flip
    return img

def train_model():
    global _model
    if _model is not None:
        return _model

    # Training data (all lines assumed black)
    static_images = [
        (2.0,  '/Users/rohanwadhwa/Documents/Black_Line_Images/2mm_Black_Line/IMG-2200.heic'),
        (2.0,  '/Users/rohanwadhwa/Documents/Black_Line_Images/2mm_Black_Line/IMG-2201.heic'),
        (2.0,  '/Users/rohanwadhwa/Documents/Black_Line_Images/2mm_Black_Line/IMG-2202.heic'),
        (2.0,  '/Users/rohanwadhwa/Documents/Black_Line_Images/2mm_Black_Line/IMG-2203.heic'),
        (20.0, '/Users/rohanwadhwa/Documents/Black_Line_Images/20_mm_Red_Line/IMG-2173.heic'),
        (20.0, '/Users/rohanwadhwa/Documents/Black_Line_Images/20_mm_Red_Line/IMG-2217.heic'),
        (40.0, '/Users/rohanwadhwa/Documents/Black_Line_Images/40_mm_Red_Line/IMG_2177.heic'),
        (47.0, '/Users/rohanwadhwa/Documents/Black_Line_Images/47_mm_Red_Line/IMG-2195.heic'),
        (52.0, '/Users/rohanwadhwa/Documents/Black_Line_Images/52_mm_Red_Line/IMG-2197.heic'),
        (56.0, '/Users/rohanwadhwa/Documents/Black_Line_Images/56_mm_Red_Line/IMG-2194.heic'),
        (60.0, '/Users/rohanwadhwa/Documents/Black_Line_Images/60_mm_Red_Line/IMG-2192.heic'),
        (60.0, '/Users/rohanwadhwa/Documents/Black_Line_Images/60_mm_Red_Line/IMG-2233.heic'),
        (60.0, '/Users/rohanwadhwa/Documents/Black_Line_Images/60_mm_Red_Line/IMG-2234.heic'),
        (60.0, '/Users/rohanwadhwa/Documents/Black_Line_Images/60_mm_Red_Line/IMG-2238.heic'),
        (63.0, '/Users/rohanwadhwa/Documents/Black_Line_Images/63_mm_Red_Line/IMG-2198.heic'),
        (67.0, '/Users/rohanwadhwa/Documents/Black_Line_Images/67_mm_Red_Line/IMG-2193.heic'),
        (80.0, '/Users/rohanwadhwa/Documents/Black_Line_Images/80_mm_Black_Line/IMG-2282.heic'),
        (80.0, '/Users/rohanwadhwa/Documents/Black_Line_Images/80_mm_Black_Line/IMG-2284.heic'),
        (80.0, '/Users/rohanwadhwa/Documents/Black_Line_Images/80_mm_Black_Line/IMG-2285.heic'),
        (80.0, '/Users/rohanwadhwa/Documents/Black_Line_Images/80_mm_Black_Line/IMG-2286.heic'),
        (80.0, '/Users/rohanwadhwa/Documents/Black_Line_Images/80_mm_Black_Line/IMG-2287.heic'),
        (90.0, '/Users/rohanwadhwa/Documents/Black_Line_Images/90_mm_Black_Line/IMG-2310.heic'),
        (90.0, '/Users/rohanwadhwa/Documents/Black_Line_Images/90_mm_Black_Line/IMG-2311.heic'),
        (90.0, '/Users/rohanwadhwa/Documents/Black_Line_Images/90_mm_Black_Line/IMG-2312.heic'),
    ]

    imgs, lens = [], []
    for mm, path in static_images:
        img_array = load_rgb(path)
        # Augment 20x for 2mm, 10x for others to balance dataset
        aug_count = 20 if mm == 2.0 else 10
        for _ in range(aug_count):
            aug_img = augment_image(img_array.copy())
            # Apply masking before resizing
            hsv = cv2.cvtColor(aug_img, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 40))  # Optimized for black lines
            clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MASK_KERNEL)
            clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, MASK_KERNEL)
            masked_img = cv2.bitwise_and(aug_img, aug_img, mask=clean)
            masked_img = cv2.resize(masked_img, IMG_SIZE)
            imgs.append(masked_img / 255.0)
            lens.append(mm)

    x = np.array(imgs, dtype=np.float32)
    y = np.array(lens, dtype=np.float32)

    # Normalize labels
    y_mean, y_std = np.mean(y), np.std(y)
    y_normalized = (y - y_mean) / y_std

    # Train-validation split
    x_train, x_val, y_train, y_val = train_test_split(x, y_normalized, test_size=0.2, random_state=SEED)

    # Reshape for MLP
    x_train = x_train.reshape((x_train.shape[0], -1))
    x_val = x_val.reshape((x_val.shape[0], -1))

    # MLP with stabilization
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE[0] * IMG_SIZE[1] * 3,)),
        layers.BatchNormalization(),  # Stabilize training
        layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.005)),
        layers.Dropout(0.4),  # Increased to reduce overfitting
        layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.005)),
        layers.Dropout(0.4),
        layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.005)),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)  # Lower learning rate
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)

    # Train
    model.fit(
        x_train, 
        y_train,
        validation_data=(x_val, y_val),
        epochs=30,
        batch_size=32,  # Increased to reduce overfitting
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    _model = model
    _model.y_mean = y_mean
    _model.y_std = y_std
    return model

def preprocess_predict_with_overlay(path, actual_mm):
    model = train_model()
    image = Image.open(path).convert("RGB")
    img_array = np.array(image).astype(np.uint8)
    
    # For prediction (keep masking as in training)
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 40))  # Same as training
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MASK_KERNEL)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, MASK_KERNEL)
    masked_img = cv2.bitwise_and(img_array, img_array, mask=clean)
    resized = cv2.resize(masked_img, IMG_SIZE)
    img_input = resized.astype(np.float32) / 255.0
    img_input = img_input.reshape(1, -1)

    # Predict and denormalize
    pred_normalized = model.predict(img_input, verbose=0)[0][0]
    pred_mm = pred_normalized * model.y_std + model.y_mean
    residual = pred_mm - actual_mm
    error_percent = abs(residual / actual_mm) * 100 if actual_mm != 0 else float('inf')

    # Use original image for annotation (resize to match prediction size)
    original_resized = cv2.resize(img_array, IMG_SIZE)
    annotated = original_resized.copy()
    text_lines = [
        f"Predicted Length: {pred_mm:.2f} mm",
        f"Actual Length:    {actual_mm:.2f} mm",
        f"Residual Error:   {residual:+.2f} mm",
        f"Error:            {error_percent:.2f}%"
    ]
    y0, dy = 20, 20
    for i, line in enumerate(text_lines):
        y = y0 + i * dy
        cv2.putText(annotated, line, (10, y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.4, color=(255, 255, 255), thickness=1)

    return annotated, pred_mm, residual
    
