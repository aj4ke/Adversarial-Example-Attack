import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Load trained model
victim_model = keras.models.load_model("mnist_victim_model.keras")

# Load MNIST
(_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
y_test = keras.utils.to_categorical(y_test, 10)
y_test_labels = np.argmax(y_test, axis=1) 

# Normalize
x_test = x_test / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)

# FGSM attack
def fgsm_attack(model, image, label, epsilon=0.3):
    image = tf.convert_to_tensor(image.reshape(1, 28, 28, 1), dtype=tf.float32)
    label = tf.convert_to_tensor(label.reshape(1, 10), dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = keras.losses.categorical_crossentropy(label, prediction)

    # Gradient and perturbation
    gradient = tape.gradient(loss, image)
    adversarial_example = image + epsilon * tf.sign(gradient)

    return tf.clip_by_value(adversarial_example, 0, 1).numpy()

selected_images = []
selected_labels = []
original_digits = []

for digit in range(10):
    idx = np.where(y_test_labels == digit)[0][0]
    selected_images.append(x_test[idx])
    selected_labels.append(y_test[idx])
    original_digits.append(digit)

selected_images = np.array(selected_images)
selected_labels = np.array(selected_labels)

# Adversarial examples
start_time = datetime.now()
adv_examples = np.array([fgsm_attack(victim_model, img, lbl) for img, lbl in zip(selected_images, selected_labels)])
end_time = datetime.now()

attack_time = (end_time - start_time).total_seconds()

# Predict original and adversarial classifications
original_preds = np.argmax(victim_model.predict(selected_images), axis=1)
adv_preds = np.argmax(victim_model.predict(adv_examples), axis=1)

# Attack success rate
misclassified = sum(original_preds != adv_preds)
success_rate = (misclassified / 10) * 100

print("\n=== Results ===")
print(f"Total Attack Time: {attack_time:.4f} seconds")
print(f"Attack Success Rate: {success_rate:.2f}%")
print("\nOriginal vs. Adversarial Predictions:")

for i in range(10):
    print(f"Orig: {original_preds[i]} -> Adv: {adv_preds[i]}")

fig, axs = plt.subplots(2, 10, figsize=(14, 5)) 

for i in range(10):
    # Original image
    axs[0, i].imshow(selected_images[i].squeeze(), cmap='gray')
    axs[0, i].set_title(f"Orig: {original_digits[i]}", fontsize=10, pad=8)
    axs[0, i].axis('off')

    # Adversarial image
    axs[1, i].imshow(adv_examples[i].squeeze(), cmap='gray')
    axs[1, i].set_title(f"Adv: {adv_preds[i]}", fontsize=10, pad=8)
    axs[1, i].axis('off')

plt.suptitle(f"FGSM Attack (ε=0.3)\nTotal Attack Time: {attack_time:.4f} sec\nSuccess Rate: {success_rate:.2f}%", fontsize=14, y=0.95)
plt.tight_layout(pad=2.5)
plt.savefig("results/fgsm_attack_summary.png", bbox_inches='tight')