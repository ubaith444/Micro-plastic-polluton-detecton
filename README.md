
Microplastic Detection in Water using CNN (Deep Learning)

📘 Project Overview

This project aims to automatically detect and classify microplastic particles in water samples using Convolutional Neural Networks (CNNs). Microplastics are tiny plastic fragments that pollute oceans, rivers, and lakes, posing serious risks to aquatic life and human health.
The proposed system provides a faster, accurate, and cost-effective way to analyze water sample images, reducing manual effort and supporting environmental sustainability.


---

🧠 Objectives

To detect and classify microplastic particles using CNN-based image analysis.

To automate water quality monitoring with minimal human intervention.

To support sustainable development by reducing plastic pollution.



---

⚙️ Technologies Used

Python

TensorFlow / Keras

OpenCV

NumPy, Pandas, Matplotlib

Google Colab / Jupyter Notebook



---

📂 Dataset

Use open-source datasets of microplastic or marine debris images from platforms like:

CleanSea & e-CleanSea Dataset

The CleanSea and e-CleanSea datasets contain real and synthetic underwater images used for detecting and classifying marine debris. They include 19 object categories such as plastic bottles, bags, cans, nets, and glass. CleanSea provides real underwater photos, while e-CleanSea offers computer-generated images to improve model training.

All images are annotated for object detection and segmentation in formats like COCO and Pascal VOC. These datasets are ideal for CNN models (YOLO, Faster R-CNN, Mask R-CNN) used in marine pollution detection. They support sustainability by promoting clean oceans and contributing to SDG 14 – Life Below Water.

Link: http://www.dlsi.ua.es/~jgallego/datasets/cleansea/


Each image should be labeled into classes such as:

Fragments

Fibers

Pellets

Non-plastic



---

🧩 Project Workflow

1. Data Collection: Gather or download labeled water sample images.


2. D Model Building: Create and train a CNN model using TensorFlow/Keras.


4. Testing: Evaluate accuracy and performance on test data.


5. Prediction: Classify and count microplastic types in new images.




---

🌱 Sustainability Impact

This project contributes to:

SDG 6: Clean Water and Sanitation

SDG 14: Life Below Water
By enabling efficient and automated detection of water pollutants, it supports cleaner water systems and helps reduce environmental damage.



---

🧾 Expected Output

Trained CNN model capable of identifying microplastic types.

Visualization of classification results and accuracy metrics.
