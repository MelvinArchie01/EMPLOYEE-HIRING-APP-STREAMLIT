#importing necessary libaries
import streamlit as st
import pandas as  pd 
#app title
st.title("EMPLOYEE HIRING APP") #assigning a title to the streamlit app

#creating a paragraph ("  ") use this to input a paragraph
st.write(''' The Employee Hiring App empowers HR professionals 
         to make informed hiring decisions by streamlining the process of 
         identifying essential factors for different job roles and departments.
          By analyzing company data and industry trends, the app provides 
         valuable insights and recommendations to optimize talent 
         acquisition and improve employee retention.                                               ''')

#importing data to the web app

perf_model = pd.read_csv("C:\\Users\\user\\OneDrive\\Desktop\\EMPLOYEE STREAMLIT APP VS CODE\\INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.csv")

# Calculate the number of employees per department
emp_count_per_dept = perf_model["EmpDepartment"].value_counts()

# Create a bar chart using Streamlit's `st.bar_chart` function
st.title("Departments In The Firm")
st.bar_chart(emp_count_per_dept)  # Streamlit will handle x-axis labels automatically

st.header("Data Preview")
st.write(perf_model.head(5)) #printing the first 5 rows

#having user slider

num_rows = st.slider("Select the number of rows", min_value = 1, max_value = len(perf_model), value = 5)
st.write("Here are the rows you have selected in the Dataset")
st.write(perf_model.head(num_rows)) #st.write is the print function in python


#finding the shape of the dataset
st.write("viewing the number of rows and columns in the dataset:", perf_model.shape)


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
encoded_columns = ['EmpDepartment', 'EmpJobRole', 'EmpEnvironmentSatisfaction','EmpWorkLifeBalance']
le_dict = {col: LabelEncoder() for col in encoded_columns}

for column in encoded_columns:
    le_dict[column].fit(perf_model[column])
    perf_model[column] = le_dict[column].transform(perf_model[column])

# Encode target variable
le_target = LabelEncoder()
perf_model['PerformanceRating'] = le_target.fit_transform(perf_model['PerformanceRating'])

X = perf_model[['EmpDepartment', 'EmpJobRole', 'EmpEnvironmentSatisfaction','EmpWorkLifeBalance', 'ExperienceYearsInCurrentRole', 'EmpLastSalaryHikePercent', 'YearsSinceLastPromotion']]
y = perf_model['PerformanceRating']



from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
model_rf = RandomForestClassifier(n_estimators=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model_rf.fit(X_train, y_train)

y_pred = model_rf.predict(X_test)

from sklearn.metrics import accuracy_score, f1_score
accuracy = accuracy_score(y_test, y_pred)
st.header("Model Evaluation Metrics")
st.write("Accuracy Using Random Forest Classifier:", accuracy)
st.write("f1 score of rf model:", f1_score(y_test, y_pred, average='weighted'))



# Assuming you have already defined le_dict, dtc, and le_target

# User input for new data
st.sidebar.write("## Enter new data for prediction")
Employee_job_role = st.sidebar.selectbox("EmpJobRole", le_dict['EmpJobRole'].classes_)
Employee_department = st.sidebar.selectbox("EmpDepartment", le_dict['EmpDepartment'].classes_)
work_life_balance = st.sidebar.selectbox("EmpWorkLifeBalance", le_dict['EmpWorkLifeBalance'].classes_)
environment_satisfaction = st.sidebar.selectbox("EmpEnvironmentSatisfaction", le_dict['EmpEnvironmentSatisfaction'].classes_)
Current_role_experience_years = st.sidebar.number_input('ExperienceYearsInCurrentRole')
Employee_last_salary_hike_percent = st.sidebar.number_input('EmpLastSalaryHikePercent')
Employee_years_since_last_promotion = st.sidebar.number_input('YearsSinceLastPromotion')



# Encode user input
encoded_input = [
    le_dict['EmpWorkLifeBalance'].transform([work_life_balance])[0],
    le_dict['EmpEnvironmentSatisfaction'].transform([environment_satisfaction])[0],
    le_dict['EmpJobRole'].transform([Employee_job_role ])[0],
    le_dict['EmpDepartment'].transform([Employee_department ])[0],
    
    Current_role_experience_years,
    Employee_last_salary_hike_percent,
    Employee_years_since_last_promotion
    
]

performance_rating_map = {
    1: "Low",
    2: "Good",
    3: "Excellent",
    4: "Outstanding"
}

# Predict using the model

#if st.sidebar.button('Performance Rating'):
    #prediction = model_rf.predict([encoded_input])[0]
    #predicted_income_group = performance_rating_map[prediction]
    #st.sidebar.write('Predicted Performance :', le_target.inverse_transform([prediction])[0])

if st.sidebar.button('Performance Rating'):
    prediction = model_rf.predict([encoded_input])[0]
    predicted_performance_rating = performance_rating_map[prediction]
    st.sidebar.write('Performance Rating:', predicted_performance_rating)

