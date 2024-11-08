# Employee Attrition Analysis and Job Role Recommendation System

## Project Overview
This project aims to identify factors influencing employee attrition and provides a recommendation system for job roles. Leveraging IBM's HR Analytics dataset, we build and evaluate several machine learning models to predict employee attrition. The project uses various visualization and clustering techniques to analyze employee data and provide actionable insights.

## Features
- **Machine Learning Models**: Multiple classifiers are used to predict employee attrition.
- **Recommendation System**: Suggests job roles based on clusters of similar attributes for employees with low attrition risk.
- **Visualization**: Graphs, heatmaps, and charts are used to illustrate key insights in the data.

## Dataset
The IBM HR Analytics Employee Attrition dataset, downloaded directly from Kaggle.

## Requirements
- Python 3.x
- Kaggle API (to download the dataset)
- Required Libraries:
    - pandas, numpy, matplotlib, seaborn, plotly
    - scikit-learn, XGBoost, LightGBM, CatBoost, PyTorch, tqdm

Install all dependencies using:
```bash
pip install -r requirements.txt
```

## Project Structure

- **Data Preprocessing**: Load data, remove redundant columns, perform feature scaling, and encode categorical variables.
- **Visualization**: Explore the dataset with bar charts, count plots, and correlation heatmaps.
- **Model Training and Evaluation**: Use `GridSearchCV` to fine-tune hyperparameters, evaluate models, and display feature importance for each model.
- **Recommendation System**: Implemented with clustering (KMeans) to suggest similar job roles based on employee attributes.

## Key Sections

### 1. Data Analysis and Visualization
- Data overview with top 5 values per feature.
- Donut charts, count plots, and bar charts to illustrate employee distribution by job role, monthly income, and overtime.

### 2. Machine Learning Models
- Models used: Logistic Regression, Decision Tree, Random Forest, CatBoost, AdaBoost, XGBoost, GradientBoosting, LightGBM.
- Each model is trained and evaluated using accuracy, balanced test score, classification reports, and confusion matrices.

### 3. Recommendation System
The recommendation system uses KMeans clustering to suggest job roles for employees based on similarity in attributes. Users can input a `JobRole` and `JobLevel` to receive a list of recommended roles.

## Usage

1. **Data Preprocessing**:
    The dataset is downloaded from Kaggle and extracted. Data preprocessing includes dummy encoding and scaling.

2. **Model Training and Evaluation**:
    Run the following function to train and evaluate all models:
    ```python
    for model_name, (model, params) in model_params.items():
        best_models[model_name] = train_and_evaluate(model, params, x_train_std, y_train, x_test_std, y_test)
    ```

3. **Get Job Recommendations**:
    Initialize the recommendation system:
    ```python
    recommender = Recommend(df)
    recommendations = recommender.get_recomm('JobRole', JobLevel)
    print(recommendations)
    ```

## Results and Evaluation
- Logistic Regression and CatBoost achieved the best results based on cross-validation accuracy and balanced test scores.
- The recommendation system effectively groups employees based on similar job attributes, providing actionable recommendations.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
