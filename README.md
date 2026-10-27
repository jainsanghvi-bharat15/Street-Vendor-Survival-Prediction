# Street Vendor Survival Prediction Using Machine Learning

Project involving analysis of the data related to the various factors which influence the survival of the street vendors in Urban India. It will be based on analysis of data using Python, feature engineering, machine learning, SQL, Power BI, and a Streamlit Web application.

> **Project Status:** Academic prototype of project. The Streamlit application allows you to explore datasets and make predictions using a trained model that is saved when the appropriate feature-engineered datasets are provided. This is not a validated system for real-world decisions.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Project Objectives](#project-objectives)
- [Dataset Description](#dataset-description)
- [Technologies Used](#technologies-used)
- [Project Workflow](#project-workflow)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Machine Learning](#machine-learning)
- [SQL Analysis](#sql-analysis)
- [Power BI Dashboard](#power-bi-dashboard)
- [Streamlit Web Application](#streamlit-web-application)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [Running the Application](#running-the-application)
- [Understanding the Prediction Demo](#understanding-the-prediction-demo)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Learning Outcomes](#learning-outcomes)
- [Author](#author)

---

## Project Overview

Street vendors are an integral part of the urban Indian economy. Their business performance is likely to be related to multiple parameters, such as daily income, consumer demand, experience in business, costs, competition, licenses, health checks, and consumer grievances.

In this project, the Street Vendor dataset is analyzed in order to understand this correlation and to show how data analytics and machine learning can be used in a real business problem.

The project follows a data-driven workflow:

1. Dataset inspection and cleaning.
2. Exploratory data analysis (EDA).
3. Feature engineering and categorical encoding.
4. Machine learning experimentation.
5. SQL-based data analysis.
6. Interactive dashboards using Power BI Desktop.
7. A local Streamlit application for data exploration and model demonstration.

The objective is to build an academic analytics prototype that brings together data preparation, analysis, visualization, and predictive modeling.

## Problem Statement

Street vendors operate under conditions where income, customer demands, rents, competition, and regulation may differ significantly.

In this study, we will examine the following research question:

**Can past data on the vendor be used to predict patterns in connection with the vendor's 12-month survival rate?**

This study examines the connections between vendor attributes and the target attribute of vendor_survived.

The findings will only serve academic purposes and not be applied to making any real business, loan, license, or regulatory decisions.

## Project Objectives

- Learn the data structure and quality of the dataset.
- Clean and preprocess the data.
- Investigate correlations between business features and survival probabilities.
- Generate extra features for machine learning.
- Test different types of classifiers.
- Query and analyze data in SQL.
- Visualize the results in Power BI.
- Make an interactive Streamlit app for vendor data exploration.
- Deploy a trained model into a prediction demo locally.
- Describe the process and limitations of the project.

## Dataset Description

The project uses an urban street food vendor survival dataset representing vendors across Indian cities.

### Dataset Details

| Attribute | Description |
|---|---|
| Dataset domain | Urban street food vendors in India |
| Number of records | 22,000 |
| Original dataset columns | 20 |
| Feature-engineered dataset columns | 64, including the target |
| Target variable | `vendor_survived` |
| Prediction type | Binary classification |
| Outcome period | 12 months |

The dataset is a simulated dataset covering the period 2019–2023. Its patterns should not automatically be treated as representative of actual street vendors.



### Cities Represented

- Mumbai
- Delhi
- Bengaluru
- Hyderabad
- Pune
- Jaipur
- Lucknow
- Kochi

### Important Features

| Feature | Description |
|---|---|
| `vendor_id` | Vendor identifier |
| `city` | City where the vendor operates |
| `zone_type` | Type of operating zone |
| `vendor_age_years` | Vendor's age in years |
| `years_in_business` | Business experience |
| `food_category` | Type of food sold |
| `license_status` | Vendor's licensing status |
| `avg_daily_revenue_inr` | Average daily revenue in INR |
| `avg_daily_customers` | Average daily customer count |
| `monthly_stall_rent_inr` | Monthly stall rent in INR |
| `num_helpers` | Number of helpers |
| `hours_open_per_day` | Daily operating hours |
| `competition_within_100m` | Competition near the vendor |
| `monthly_health_inspection_score` | Recorded health inspection score |
| `had_fine_last_year` | Whether a fine was recorded in the previous year |
| `avg_monthly_rainfall_mm` | Average monthly rainfall |
| `season_of_observation` | Season of observation |
| `has_online_presence` | Whether the vendor has an online presence |
| `customer_complaint_rate` | Customer complaint rate |
| `vendor_survived` | Recorded 12-month outcome |

### Target Variable

The target variable is `vendor_survived`.

| Value | Meaning |
|---|---|
| `1` | Survived |
| `0` | Closed |

The dataset contains:

- **9,364 survived records (42.56%)**
- **12,636 closed records (57.44%)**

These figures describe the supplied dataset, not the current status of real-world vendors.

### Data Quality

The dataset has several missing values in different attributes, such as health inspection score, customer complaints, revenues, number of customers, and rent.

It is important to validate the missing values and types of data before analyzing and modeling. The missing value handling must be recorded so that the model input attributes do not change.

---

## Technologies Used

### Programming and Data Analysis

- **Python** : Main programming language.
- **Pandas** : Data manipulation and analysis.
- **NumPy** : Numerical operations.
- **Matplotlib** : Data visualization.
- **Seaborn** : Exploratory visualizations.
- **Scikit-learn** : Machine learning utilities and evaluation.
- **XGBoost** : Gradient-boosted classification model.

### Database

- **MySQL** : Structured data storage and SQL analysis.

### Visualization

- **Power BI Desktop** : Interactive analytical dashboards.

### Application Development

- **Streamlit** : Local interactive web application.
- **Joblib and Pickle** : Loading saved model artifacts, depending on how the model was serialized.
- **VS Code / Jupyter Notebook** : Development and experimentation environments.

---

## Project Workflow

The project follows these stages:

### 1. Data Collection and Inspection

- Load the dataset.
- Inspect the number of records and columns.
- Check data types and missing values.
- Examine the target variable distribution.
- Identify fields relevant to vendor business characteristics.

### 2. Data Cleaning

- Inspect missing and inconsistent values.
- Prepare numerical and categorical fields.
- Verify the target variable.
- Save a cleaned dataset for subsequent analysis.

### 3. Exploratory Data Analysis

- Analyze the distribution of numerical features.
- Compare vendor outcomes across categories.
- Examine relationships between revenue, customer demand, competition, and survival.
- Generate visualizations to identify patterns.

### 4. Feature Engineering

- Create additional business-related features.
- Encode categorical variables.
- Prepare the feature matrix for machine learning.
- Keep the training and prediction feature definitions consistent.

### 5. Machine Learning

- Prepare model inputs and the target.
- Train and evaluate classification models.
- Save the trained model for reuse.
- Use the saved model in the Streamlit prediction demonstration.

### 6. SQL Analysis

- Store and query structured data in MySQL.
- Explore records using filtering, grouping, aggregation, and other SQL operations.

### 7. Power BI Visualization

- Build dashboard pages for business overview, analysis, and risk-related insights.
- Present important metrics and patterns through interactive visualizations.

### 8. Streamlit Application

- Display dataset summaries.
- Allow users to filter vendor records.
- Load a saved machine learning model.
- Demonstrate predictions using compatible feature-engineered records.

---

## Exploratory Data Analysis

Exploratory Data Analysis was used to understand the dataset and investigate relationships between vendor characteristics.

The analysis considered:

- Distribution of survival outcomes.
- Vendor records across cities.
- Food category distribution.
- Revenue and customer demand.
- Vendor age and years in business.
- Health inspection scores.
- Customer complaints.
- Competition near the vendor.
- Operating expenses and business characteristics.

### Observed Patterns

The exploratory analysis indicated several relationships in the supplied dataset:

- Daily revenue and daily customer counts showed a positive correlation.
- Vendor age and years in business showed a positive relationship.
- Revenue and customer demand showed modest positive associations with the survival target.
- Customer complaint rate and nearby competition showed weak negative associations with the survival target.

These are descriptive associations, not evidence that any particular factor causes a vendor to survive or close.

The findings may also reflect the way the simulated dataset was generated.

---

## Feature Engineering

Feature engineering was used to create additional variables that may help describe vendor operations and business performance.

Examples of derived features include:

| Feature | Purpose |
|---|---|
| `estimated_monthly_revenue` | Estimates monthly revenue from daily revenue |
| `revenue_per_customer` | Examines revenue relative to customer count |
| `rent_to_revenue_ratio` | Relates stall rent to estimated revenue |
| `revenue_per_hour` | Examines revenue relative to operating hours |
| `customers_per_hour` | Examines customer volume relative to operating hours |
| `operational_load` | Represents a derived measure of operational activity |

The project also explored categorical transformations for variables such as revenue, age, experience, competition, inspection scores, and complaint rates.

Categorical encoding, including one-hot encoding, was used to prepare features for modeling.

**Important:** The exact feature definitions, encoding rules, column order, and missing-value treatment used during training must be preserved when applying the trained model to new data.

---

## Machine Learning

The project treats vendor survival as a binary classification problem.

### Target

`vendor_survived`

- `1`: Survived
- `0`: Closed

### Model

The saved model used by the current application is `trained_model.pkl`, identified as an XGBoost classification model.

### Model Workflow

1. Prepare the dataset.
2. Separate input features from the target.
3. Apply the training-time feature engineering and encoding.
4. Train a classification model.
5. Evaluate the model using suitable classification metrics.
6. Save the trained model.
7. Load the saved model in the Streamlit application.
8. Generate predictions from compatible feature-engineered records.

### Evaluation

Model performance should be reported using results from a documented evaluation procedure, preferably including a held-out test set.

Useful classification metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

**No model accuracy is claimed in this README.** Add your verified evaluation results after confirming the actual test-set metrics from your training notebook or evaluation report.

---

## SQL Analysis

MySQL is used to organize and query vendor data.

SQL analysis can help answer questions such as:

- How many vendor records are available for each city?
- Which food categories have the most records?
- What is the distribution of recorded survival outcomes?
- How do average revenue and customer counts vary across groups?
- How do business characteristics differ across categories?

Typical SQL operations include:

- `SELECT`
- `WHERE`
- `GROUP BY`
- `ORDER BY`
- Aggregate functions such as `COUNT()`, `AVG()`, `MIN()`, and `MAX()`
- Joins where multiple related tables are available

The SQL scripts in the repository should be consulted for the exact queries and database structure implemented in the project.

---

## Power BI Dashboard

The project uses Power BI Desktop to present analytical findings.

The planned dashboard organization consists of three pages.

### Page 1: Executive Overview

Provides a high-level summary of the dataset and its recorded outcomes.

Potential components include:

- Total vendor records.
- Survived and closed counts.
- Outcome distribution.
- City-wise vendor distribution.
- Food category distribution.

### Page 2: Business Analysis

Explores business characteristics and operational patterns.

Potential components include:

- Revenue and customer demand.
- Vendor experience.
- Operating expenses.
- Food category comparisons.
- City-wise and zone-wise analysis.

### Page 3: Vendor Risk and Operational Insights

Explores patterns associated with the recorded outcome.

Potential components include:

- Competition.
- Health inspection scores.
- Customer complaints.
- Licensing status.
- Survival outcome comparisons.

Dashboard descriptions should be adjusted to match the visuals and measures actually present in the saved Power BI report.

**Note:** Power BI Desktop is used for report development. Publication to Power BI Service is not assumed.

---

## Streamlit Web Application

The Streamlit application provides a local interface for interacting with the project dataset and saved machine learning model.

### 1. Dataset Overview

Displays:

- Total number of records.
- Survived and closed counts.
- Recorded survival rate.
- Outcome distribution.
- City-wise vendor counts.
- Food category distribution.

### 2. Explore Vendors

Allows users to:

- Filter records by city.
- Filter by zone type.
- Filter by food category.
- Filter by licensing status.
- Filter by recorded outcome.
- Inspect selected vendor attributes.
- View numerical summaries.

The application displays up to 1,000 matching records in the main table for readability.

### 3. Prediction Demo

Prediction Page:

- Loads the already saved trained model.
- Validates the feature names for which the model was created.
- Looks for a CSV that has the model features.
- Enables selection of an already available vendor record.
- Passes the feature engineered value of the record to the model.
- Shows the predicted class.
- Shows class probabilities wherever possible in the model.
- Considers the predicted value against the stored target where possible.

The current implementation works on the existing dataset records instead of using an independent form to input new vendors.

### Model and Dataset Requirements

The application checks the following model locations:

```text
trained_model.pkl
models/trained_model.pkl
models/vendor_survival_model.joblib
```

Place the feature-engineered CSV files inside:

```text
datasets/processed_dataset/
```

The chosen CSV file needs to have all the features that the model requires.

If the model had been trained using 63 input features, then the prediction dataset will have to supply the same number of input features.

A CSV file having only the columns from the original clean CSV file will be inadequate.

### Prediction Interpretation

This prediction is only an output of an experimental model. The prediction cannot ensure the survival or demise of the vendor.

Any prediction of records used for training a model is not indicative of performance on unseen data.

---

## Project Structure

The project repository may be organized as follows:

```text
Street-Vendor-Survival-Prediction/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── datasets/
│   ├── raw_dataset/
│   └── processed_dataset/
│
├── notebooks/
│   ├── data_cleaning.ipynb
│   ├── exploratory_data_analysis.ipynb
│   ├── feature_engineering.ipynb
│   └── model_training.ipynb
│
├── models/
│   ├── trained_model.pkl
│   └── model_features.json
│
├── sql/
│   └── analysis_queries.sql
│
├── powerbi/
│   └── vendor_survival_dashboard.pbix
│
├── figures/
│
└── reports/
```

It is an example structure. Include only those files and folders that are present in your repository. Rename the files to reflect the artifacts you have saved.

Don’t commit large data sets or model files on GitHub without verifying their size, licensing, and repository requirements.

---

## Installation and Setup

### Prerequisites

Install the following:

- Python 3.10 or another version compatible with your model dependencies.
- pip.
- Git, if you want to clone the repository.
- MySQL, if you want to run the SQL analysis.
- Power BI Desktop, if you want to open the dashboard.

The Python version and library versions used during model training are important for loading a serialized model successfully.

### 1. Clone the Repository

Replace the repository URL below with your actual GitHub repository URL if necessary.

```bash
git clone https://github.com/jainsanghvi-bharat15/Street-Vendor-Survival-Prediction.git
```

### 2. Open the Project Directory

```bash
cd Street-Vendor-Survival-Prediction
```

### 3. Create a Virtual Environment

On Windows:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run the commands using the Python executable inside `.venv` instead.

### 4. Install Dependencies

If `requirements.txt` is available:

```bash
python -m pip install -r requirements.txt
```

Otherwise, install the main application packages:

```bash
python -m pip install streamlit pandas numpy scikit-learn xgboost joblib
```

If your saved model was created using specific package versions, install compatible versions from the original training environment.

### 5. Add the Required Files

Ensure the project contains:

- `app.py`
- The cleaned CSV dataset.
- The feature-engineered CSV used for model training.
- The saved trained model.

The app expects the datasets inside:

```text
datasets/processed_dataset/
```

The trained model can be placed beside `app.py` as `trained_model.pkl`, or inside the `models/` directory.

---

## Running the Application

From the project root, run:

```bash
python -m streamlit run app.py
```

Streamlit will display a local address in the terminal, usually:

```text
http://localhost:8501
```

Open the URL in your web browser.

Use the sidebar to move from:

1. Dataset Overview
2. Explore Vendors
3. Prediction Demo

If the application is unable to detect any datasets, ensure that the CSV filenames and the `datasets/processed_dataset/` folder are correct.

If the prediction module is unable to detect any compatible features, ensure that the processed dataset is available.

---

## Understanding the Prediction Demo

The prediction demo aims at illustrating how a previously developed classification model can be included into a data analytics application.

In order to ensure validity of predictions, the following should correspond to the initial training process:

- Input feature names.
- Feature order, where required by the model.
- Numerical data types.
- Categorical encoding.
- Feature engineering formulas.
- Missing-value treatment.
- Any scaling or transformations applied during training.

Just the saved classifier may not have all preprocessing steps contained in it.

If the classifier was trained with a preprocessing pipeline, then that also needs to be used to make any predictions.

Currently, the application tests for compatible features, but just matching column names does not mean that each step in the preprocessing pipeline is identical to that of training.

---

## Limitations

1. **Simulated dataset:** The dataset does not establish that the observed relationships hold for actual street vendors.
2. **Recorded outcome:** The target represents a 12-month dataset outcome, not a verified current business status.
3. **Model validation:** Performance must be established using a suitable held-out test set.
4. **Preprocessing dependency:** Predictions may be invalid if the input features differ from those used during training.
5. **Existing-record demonstration:** The current Streamlit prediction page uses existing feature-engineered records rather than a fully implemented new-vendor input form.
6. **No causal conclusions:** Correlations and feature relationships do not establish that a variable causes survival or closure.
7. **No real-world decision use:** The prototype should not be used to make lending, licensing, financial, or policy decisions.
8. **No live data integration:** The application uses local files and does not automatically obtain current vendor information.
9. **Security of serialized models:** Only load trusted model files because Pickle and Joblib artifacts can execute code during deserialization.

---

## Future Improvements

Potential improvements include:

- Validate the models with the help of a completely separate test dataset.
- Compare the classification models by using suitable measures.
- Make the data preprocessing consistent by saving the entire process of the training process.
- Incorporate a vendor form that is new after replicating the same training process accurately.
- Include model evaluation graphs and confusion matrix.
- Employ techniques of explainability like SHAP, wherever possible.
- Enhance the dashboard navigation and filtering process.
- Link the application to the database.
- Include input validation.
- Perform a performance check with real-world data first.

These are proposed improvements, not claims about features already completed.

---

## Learning Outcomes

Through this project, you get hands-on practice with:

- Data cleaning and preparation.
- Exploratory data analysis.
- Feature engineering.
- Classification modeling.
- Model serialization and deserialization.
- SQL queries and organizing data.
- Power BI dashboard creation.
- Streamlit app building.
- Machine learning in the context of analytics.
- The weaknesses of predictive models.

---

## Author

**Bharat Jain Sanghvi**

B.Tech in Computer Science and Business Systems (CSBS)  
Institute of Engineering and Technology, DAVV, Indore

### Project Focus

Data Analytics | Machine Learning | SQL | Power BI | Python | Streamlit

---

## Disclaimer

The purpose of this project was to help students learn academically and demonstrate their knowledge. The dataset used in this case is artificial, and the predictions made by models are experimental. Therefore, these predictions should not be used to make any decisions about real-life street vendors.
