# Object Detection: Raw Image Data Analysis and Preprocessing, Custom Architecture vs. Transfer Learning

This project comes from the motivation of developing an End-to-end Object detection pipeline with custom vs. industry standard architectures. Using the **PASCAL VOC 2007** dataset, it demonstrates the inner complexity of SOTA models, the abstraction and usability of open-source frameworks, and the trade-offs between custom architecture design and using pre-trained foundational models.

---

## Structure:

* `1-image-data-analysis-and-preprocessing.ipynb`: where I explored and analyzed the **PASCAL VOC 2007** dataset, a foundational dataset in computer vision. After that, a full preprocessing pipeline is implemented to parse the raw **XML annotations**, achieving a functional structure suitable for YOLO architectures.

<div align="center">
  <img width="1340" alt="image" src="https://github.com/user-attachments/assets/036d57e9-3c74-4ed8-b2b3-b720be1f5d96">
  <br>
  <em>PASCAL VOC 2007 sample preview</em>
</div>

* `2-custom-architecture-from-scratch.ipynb`: I implemented a simplified **YOLOv3** architecture to understand the core mechanics of one stage detectors. To achieve that I built from scratch a Custom Darknet-style backbone with Residual Blocks and Feature Pyramid Network (FPN), to then include manual implementation of **IoU (Intersection over Union)** and Multi-part Loss (Box + Objectness + Class). I also made a custom visual validation pipeline to compare validation samples vs. how the model performed with **NMS (Non-Max Suppression)**.

<div align="center">
  <img width="1168" height="824" alt="image" src="https://github.com/user-attachments/assets/975d031a-538f-4e1d-b062-fe90ebd18056" />
  <br>
  <em>Custom Architecture Design</em>
</div>

* `3-object-detection-with-transfer-learning.ipynb`: This is the final part, where I implemented and fine-tuned the **YOLOv8** architecture from [Ultralytics](https://www.ultralytics.com/) to obtain industry standard performance and compare it to the custom model from the previous stage. I achieved a **0.674 mAP@50**, effectively solving the class-imbalance issues found in the custom implementation.

---

## Benchmark Comparison:

<div align="center">

| Metric | Mini-YOLO (From Scratch) | YOLOv8 (Transfer Learning) |
| :--- | :--- | :--- |
| **Architecture** | Shallow Custom CNN | Deep CSPDarknet |
| **Training Time** | ~1.15 hours | ~17 minutes|
| **Loss Trend** | High Variance (Overfitting) | Stable Convergence |
| **Detection Quality** | Good Localization, Bad Classification | Perfect Localization & Classification |
| **mAP@50** | N/A (Visual Validation Only) | 67.4% |

</div>

<br>

<table align="center">
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/84b70050-38ee-4da7-9b75-7fc77d42953d" width="400" />
      <br />
      <em>Mini-YOLO (Custom)</em>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/70c7a2a6-6522-4b5a-8a20-b2a72021327f" width="400" />
      <br />
      <em>YOLOv8 (Transfer Learning)</em>
    </td>
  </tr>
</table>

---

### Tech Stack:
* **Deep Learning**: PyTorch, Ultralytics YOLOv8
* **Data Processing**: OpenCV, NumPy, Pandas, XML (ElementTree)
* **Visualization**: Matplotlib, Seaborn

### Acknowledgements:
* The data source, [PASCAL VOC 2007 Challenge](https://www.kaggle.com/datasets/stpeteishii/pascal-voc-2007-dataset).
* Kaggle's user [Olly Powell](http://kaggle.com/ollypowell), that inspired the EDA and Preprocessing by his own work in notebooks with similar purpose.
* Redmon, Joseph and Ali Fahardi for [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640), [YOLOv3](https://arxiv.org/pdf/1804.02767) papers.
* [Geeksforgeeks' blog](https://arxiv.org/pdf/1804.02767) for the inspiration in implementing YOLOv3 from scratch.
* [Andrew Ng from DeepLearning.ai](https://www.deeplearning.ai/courses/deep-learning-specialization/) for the intuition and theory explanation of YOLO architecture. 
* [Ultralytics](https://docs.ultralytics.com/es/models/yolov8/#what-is-yolov8-and-how-does-it-differ-from-previous-yolo-versions) for the YOLOv8 framework and implementation.
