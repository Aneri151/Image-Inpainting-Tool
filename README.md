# 🖼️ Image Inpainting using Traditional Methods (PDE-Based)

## 📌 Project Overview
This project focuses on **image inpainting**, which is the process of filling missing or damaged regions in digital images.  
We implement **traditional image processing techniques** using **Partial Differential Equations (PDEs)** instead of deep learning or machine learning.

The goal is to restore images by smoothly propagating surrounding pixel information into missing areas while preserving edges and structures.

---

## 🎯 Objectives:
- Understand and implement **diffusion-based inpainting**
- Apply **PDE-based methods** for image restoration
- Remove objects, scratches, and text from images
- Compare results using different diffusion approaches

---

## 🧠 Methodology:
The project follows a **diffusion-based PDE approach**, where:
- Missing regions are filled by **interpolating pixel values**
- **Isophote (edge) directions** are extended into the damaged region
- Image information spreads from known areas to unknown areas

### 🔄 Steps:
1. Load input image
2. Create mask for missing/damaged region
3. Initialize missing pixels
4. Apply diffusion (iterative process)
5. Update pixel values using neighbors
6. Repeat until smooth filling is achieved
7. Output restored image

---

## 📥 Input
- **Input Image** → Image with damaged or unwanted region  
- **Mask Image** → Binary image:
  - White → region to fill
  - Black → region to keep

---

## 📤 Output
- Restored image with missing regions filled naturally

---

## ⚙️ Technologies Used
- Python
- NumPy
- Matplotlib (for visualization)

❌ No use of:
- Machine Learning
- Deep Learning
- OpenCV (optional, avoided as per project constraint)

---

## 📊 Applications
- Old photo restoration
- Object removal
- Scratch and crack repair
- Text removal from images

---





This project is developed as part of academic coursework to understand **traditional image processing techniques** for image restoration.

---
