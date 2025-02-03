# **School Dropout Analysis using Machine Learning & Artificial Neural Networks (ANN)**

## **1. Introduction**

School dropout is a significant issue affecting education systems in India. This project predicts whether a student is likely to drop out based on **demographic, economic, and social factors**.

### **Key Features:**

**Multiple Models** – Implements **Logistic Regression, Random Forest, XGBoost, and ANN**.  
**Best Model Selection** – Automatically chooses the highest accuracy model.  
**Data Preprocessing** – Handles missing values, encodes categorical data, and scales numerical features.

---

## **2. Dataset Description**

The dataset contains the following attributes:

| Feature                  | Description                                   |
| ------------------------ | --------------------------------------------- |
| **School**               | Name of the school                            |
| **Area**                 | Urban/Rural                                   |
| **Gender**               | Male/Female                                   |
| **Caste**                | General, SC/ST, OBC, etc.                     |
| **Age**                  | Student's age                                 |
| **Religion**             | Hindu, Muslim, Christian, etc.                |
| **Annual Family Income** | Total household income                        |
| **Dropout**              | Target variable (0 = No dropout, 1 = Dropout) |

---

## **3. Installation**

### **Prerequisites**

Ensure you have **Python 3.8+** installed on your system.

### **Steps to Install**

1 Clone the repository:

```bash
git clone https://github.com/your-username/school-dropout-analysis.git
cd school_dropout_analysis
```

2 Install required dependencies:

```bash
pip install -r requirements.txt
```

---

## **4. Usage**

To run the project and train the models, execute the following command:

```bash
python main.py
```


---

## **5. Technologies Used**

| Technology             | Purpose                             |
| ---------------------- | ----------------------------------- |
| **Python**             | Programming language                |
| **TensorFlow & Keras** | Building the ANN model              |
| **Scikit-Learn**       | Implementing traditional ML models  |
| **XGBoost**            | High-performance boosting algorithm |
| **Pandas & NumPy**     | Data processing and manipulation    |

---

## **6. Model Selection & Evaluation**

This project trains multiple models and selects the best one based on accuracy. The models used are:

**Logistic Regression** – A simple baseline classifier.  
**Random Forest** – An ensemble learning method for better generalization.  
**XGBoost** – A powerful gradient-boosting model.  
**Artificial Neural Network (ANN)** – A deep learning model trained with TensorFlow/Keras.

The model with the **highest accuracy** on the test set is **automatically selected**.

---

## **7. Future Enhancements**

**Hyperparameter Tuning** – Optimize ANN layers and ML models for better performance.  
**Feature Engineering** – Incorporate additional factors like attendance records and test scores.

---

## **8. Contributing**

Contributions are welcome! To contribute:

1. Fork this repository.
2. Create a new feature branch:
   ```bash
   git checkout -b feature-branch
   ```
3. Commit your changes and push:
   ```bash
   git commit -m "Added new feature"
   git push origin feature-branch
   ```
4. Open a pull request.

---

## **9. License**

This project is released under the **MIT License**. You are free to modify and distribute it.

---
