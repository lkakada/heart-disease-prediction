# ❤️ Heart Disease Prediction App

A Streamlit-based web application that predicts the likelihood of heart disease based on patient health data.  
Built with **Python**, **Scikit-learn**, and **Streamlit**, this app provides comprehensive visual insights into major medical risk factors and real-time predictions.

---

## 📌 Features

-   🧠 **Machine Learning Prediction** — Uses Logistic Regression with standardized features to estimate heart disease probability
-   📊 **Interactive Data Visualizations** — Six comprehensive charts including correlation heatmaps, box plots, scatter plots, and histograms
-   🖥 **Real-time User Input Form** — Sidebar interface to adjust 13 health parameters and get instant predictions
-   📈 **Medical Risk Factor Analysis** — In-depth analysis of cholesterol, blood pressure, chest pain type, fasting blood sugar, and more
-   🎯 **Color-coded Prediction Display** — Clean, professional result boxes showing risk percentage with visual indicators
-   📋 **Dataset Preview** — View and validate the training data with detailed column descriptions
-   🔍 **Feature Correlation Analysis** — Understand relationships between different health indicators

---

## 📂 Project Structure

```
heart-disease-prediction/
│
├── app.py              # Main Streamlit application
├── heart.csv           # Dataset (UCI Heart Disease dataset)
├── requirements.txt    # Python dependencies
└── README.md          # Project documentation
```

## 🚀 Installation & Setup

### 1️⃣ Clone this repository

```bash
git clone https://github.com/lkakada/heart-disease-prediction.git
cd heart-disease-prediction
```

### 2️⃣ Create a virtual environment

```bash
python -m venv venv
```

### 3️⃣ Activate the virtual environment

```bash
# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 4️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 5️⃣ Run the Application

```bash
streamlit run app.py
```

After starting, Streamlit will display a local URL in the terminal:

```
Local URL: http://localhost:8501
```

Open this URL in your web browser to access the application.

---

## 🖥️ How to Use the App

1. **View Dataset Preview**: The main page displays the first few rows of the training data with detailed column descriptions
2. **Adjust Patient Parameters**: Use the sidebar controls to input patient health data:
    - Age slider
    - Gender selection
    - Chest pain type dropdown
    - Blood pressure and cholesterol sliders
    - Additional medical indicators
3. **Get Prediction**: The app automatically updates the prediction as you change parameters
4. **Explore Visualizations**: Scroll down to view six different charts showing relationships between health factors and heart disease
5. **Interpret Results**: Color-coded results show probability percentages with clear risk indicators

---

## 📊 Dataset Information

The app uses the **UCI Heart Disease dataset**, which contains 303 patient records with 14 attributes each.

### Input Features:

| Column       | Description                                    | Values                                                                          |
| ------------ | ---------------------------------------------- | ------------------------------------------------------------------------------- |
| **age**      | Patient age in years                           | 29-77 years                                                                     |
| **sex**      | Gender                                         | 1 = Male, 0 = Female                                                            |
| **cp**       | Chest pain type                                | 0 = Typical angina, 1 = Atypical angina, 2 = Non-anginal pain, 3 = Asymptomatic |
| **trestbps** | Resting blood pressure                         | mm Hg                                                                           |
| **chol**     | Serum cholesterol                              | mg/dL                                                                           |
| **fbs**      | Fasting blood sugar > 120 mg/dL                | 1 = Yes, 0 = No                                                                 |
| **restecg**  | Resting electrocardiogram results              | 0-2 scale                                                                       |
| **thalach**  | Maximum heart rate achieved                    | Beats per minute                                                                |
| **exang**    | Exercise-induced angina                        | 1 = Yes, 0 = No                                                                 |
| **oldpeak**  | ST depression induced by exercise              | Relative to rest                                                                |
| **slope**    | Slope of peak exercise ST segment              | 0-2 scale                                                                       |
| **ca**       | Number of major vessels colored by fluoroscopy | 0-3                                                                             |
| **thal**     | Thalassemia status                             | 1-3 scale                                                                       |

### Target Variable:

-   **target** — 1 = Heart disease present, 0 = No heart disease

---

## 🤖 Machine Learning Model

The application implements a **Logistic Regression** model with the following pipeline:

1. **Data Preprocessing**: StandardScaler normalizes all features to ensure equal weighting
2. **Model Training**: Logistic regression trained on the entire dataset
3. **Prediction**: Returns both binary classification and probability scores
4. **Real-time Updates**: Model predictions update instantly as parameters change

### Model Performance Features:

-   **Standardized Features**: All inputs are scaled for optimal performance
-   **Probability Output**: Shows percentage likelihood of heart disease
-   **Visual Feedback**: Color-coded results for easy interpretation

---

## 📈 Visualizations

The app provides six comprehensive charts:

1. **Correlation Heatmap** — Shows relationships between all features
2. **Cholesterol Box Plot** — Compares cholesterol levels by heart disease status
3. **Age vs Max Heart Rate Scatter** — Plots age against maximum heart rate achieved
4. **Blood Pressure Histogram** — Distribution of resting blood pressure values
5. **Chest Pain Type Bar Chart** — Heart disease prevalence by chest pain type
6. **Fasting Blood Sugar Bar Chart** — Compares heart disease rates by blood sugar levels

---

## 🛠️ Technologies Used

-   **Python 3.x**
-   **Streamlit** — Web application framework
-   **Pandas** — Data manipulation and analysis
-   **Scikit-learn** — Machine learning model and preprocessing
-   **Matplotlib** — Data visualization
-   **Seaborn** — Statistical data visualization
-   **NumPy** — Numerical computing

---

## ⚠️ Disclaimer

**This application is for educational and demonstration purposes only.**

-   It should **NOT** be used for actual medical diagnosis or treatment decisions
-   Always consult qualified healthcare professionals for medical advice
-   The predictions are based on a limited dataset and should not replace professional medical evaluation
-   This tool is designed to demonstrate machine learning concepts in healthcare applications

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📞 Contact

**Linus Kakada** - [@lkakada](https://github.com/lkakada)

Project Link: [https://github.com/lkakada/heart-disease-prediction](https://github.com/lkakada/heart-disease-prediction)

---

## � Acknowledgments

-   [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease) for the Heart Disease dataset
-   Streamlit team for the excellent web framework
-   Scikit-learn developers for the machine learning tools

---

## �📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
