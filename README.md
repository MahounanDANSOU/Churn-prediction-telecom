# 📊 Customer Churn Prediction - Telecom Sector

This project was carried out as part of a **remote internship** with [SaiKet Systems](https://saiket.in/internship-program/)  

Objective: **analyze and predict customer churn** to help the marketing team reduce customer loss.

---

## 📁 Project Structure

📂 churn-prediction-telecom
│-- README.md
│-- requirements.txt
│-- churn_analysis.ipynb
│-- data_preparation.ipynb
│-- model_training.ipynb
│-- /data
│-- /outputs
│-- /docs


---

## 🎯 Project Goals

1. **Understand the churn phenomenon** in telecom customers  
2. **Identify key factors** leading to customer departure  
3. **Build a predictive model** to anticipate churn  
4. **Provide actionable recommendations** for retention

---

## 📊 Workflow

### 1. Data Preparation
- Loaded the dataset (`7,043 customers`)
- Handled missing values
- Encoded 16 categorical variables
- Stratified train/test split

### 2. Exploratory Data Analysis (EDA)
- Used Python libraries (`matplotlib`, `seaborn`) for visualization
- Found that:
  - Customers with month-to-month contracts had **+42% churn risk**
  - Customers with <12 months tenure had **+65% churn risk**

### 3. Customer Segmentation
- K-Means clustering on selected variables
- Identified **4 customer segments**  
- Highlighted **847 high-value customers at risk**

### 4. Predictive Modeling
- Tested **Logistic Regression**, **Decision Tree**, and **Random Forest**
- Tuned hyperparameters with `GridSearchCV`
- Best model: **ROC-AUC = 0.84**

### 5. Recommendations
- Target high-risk segments with loyalty offers
- Promote yearly contracts to reduce churn

---

## 🛠️ Tools & Libraries

- **Python**: pandas, numpy, scikit-learn, matplotlib, seaborn
- **Jupyter Notebook**
- **Git & GitHub** for version control

---

## 📈 Results

| Model              | Accuracy | ROC-AUC |
|--------------------|----------|---------|
| Logistic Regression| 80%      | 0.81    |
| Decision Tree      | 78%      | 0.79    |
| Random Forest      | **84%**  | **0.84**|

---

## 📬 Contact

If you'd like to discuss the project or my work, feel free to connect:  
- **LinkedIn**: [Your LinkedIn Link](https://linkedin.com/in/yourprofile)
- **Email**: your.email@example.com

---

## 📌 Note

This project was done as part of a **remote internship** and aimed at improving my skills in data analysis, visualization, and machine learning.

