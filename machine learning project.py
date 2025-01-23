#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

df=pd.read_csv("Top_Highest_Openings.csv")

# Step 1: Handle Missing Values
  # Drop rows with missing values, or use df.fillna() to impute missing values

# Step 2: Handle Duplicates
df = df.drop_duplicates()  # Remove duplicate rows if they exist
df.head()


# In[6]:


df.drop(['Release','Total Gross','% of Total','Average'],axis='columns',inplace=True)
df.head()


# In[7]:


from matplotlib import pyplot as plt
plt.scatter(df.Theaters,df.Distributor,marker='.',color='red')


# In[8]:


round((df.isnull().sum()/df.shape[0])*100,2)


# In[9]:


df = df.drop(columns='Date')
df = df.fillna(df.Theaters.mean())
 
df.isnull().sum()


# In[10]:


round((df.isnull().sum()/df.shape[0])*100,2)


# In[119]:


#Convert Distributor column
dummies = pd.get_dummies(df['Distributor'])

df = df.drop('Distributor',axis=1)
df.head(5)

df = pd.concat([df, dummies], axis=1)
df.head()


# In[104]:


from sklearn.preprocessing import MinMaxScaler

# Normalize the dataset
scaler_trainx = MinMaxScaler()
scaler_trainy = MinMaxScaler()
scaler_testx = MinMaxScaler()
scaler_testy = MinMaxScaler()


# In[105]:


#'target' is the column you want to predict
target = df['Opening']
df = df.drop('Opening', axis=1)

#X = scaled_df.drop('Opening', axis=1)


# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=42)

X_train =scaler_trainx.fit_transform(pd.DataFrame(X_train))
y_train =scaler_trainy.fit_transform(pd.DataFrame(y_train))
X_test =scaler_testx.fit_transform(pd.DataFrame(X_test))
y_test =scaler_testy.fit_transform(pd.DataFrame(y_test))


# In[106]:


from sklearn.tree import DecisionTreeRegressor



# Creating a Decision Tree regression model with max_depth=10 and min_samples_split=5
model = DecisionTreeRegressor(max_depth=100, min_samples_split=6, random_state=42)

# Training the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
#print("Mean Squared Error:", mse)
print(mse)
print("Prediction before denormalization:", y_pred)
print("Predictions after denromalization:", scaler_testy.inverse_transform([y_pred]))



# In[107]:


from sklearn.metrics import accuracy_score

model.score(X_test,y_test)


# In[108]:


from sklearn.linear_model import LinearRegression


# Creating a Linear Regression model
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
#print("Prediction before denormalization:", y_pred)


data = pd.DataFrame({
    'Theaters': [4200],
    '-': 0,
    '20th Century Studios': 0,
    'Columbia Pictures': 0,
    'Dimension Films': 0,
    'DreamWorks': 0,
     'DreamWorks Distribution': 0,
     'FilmDistrict': 0,
     'Focus Features': 0,
     'Lionsgate Films': 0,
     'Metro-Goldwyn-Mayer': 0,
     'Miramax': 0,
     'New Line Cinema': 0,
     'Newmarket Films': 0,
     'Paramount Pictures': 0,
     'Relativity Media': 0,
     'Revolution Studios': 0,
     'STX Entertainment': 0,
     'Screen Gems': 0,
     'Sony Pictures Releasing': 0,
     'Summit Entertainment': 0,
     'The Weinstein Company': 0,
     'TriStar Pictures': 0,
     'Twentieth Century Fox': 0,
     'United Artists Releasing': 0,
     'Universal Pictures': 0,
     'Universal Pictures International': 0,
     'Walt Disney Studios Motion Pictures': 0,
     'Warner Bros.': 1
    })


y_testing = model.predict(data)
print(scaler_testy.inverse_transform(y_testing))
print("Predictions after denromalization:", scaler_testy.inverse_transform(y_pred))


# In[ ]:





# In[75]:


from sklearn.metrics import accuracy_score
model.score(X_test,y_test)


# In[76]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Creating a Random Forest regression model
model = RandomForestRegressor(random_state=42)

# Training the model
model.fit(X_train, y_train.ravel())  # or y_train.squeeze()

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test.ravel(), y_pred)  # or y_test.squeeze()
print("Mean Squared Error:", mse)

#print("Prediction before denormalization:", y_pred)
print("Predictions after denormalization:", scaler_testy.inverse_transform([y_pred]))


# In[77]:


from sklearn.metrics import accuracy_score
model.score(X_test,y_test)


# In[2]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import messagebox


# Load the dataset
df = pd.read_csv("Top_Highest_Openings.csv")

df = df.fillna(df.Theaters.mean())

df.isnull().sum()

# Handle Duplicates
df = df.drop_duplicates()

# Convert Distributor column
dummies = pd.get_dummies(df['Distributor'])
df = pd.concat([df, dummies], axis=1)
df.drop(['Release','Date','Total Gross','% of Total','Average','Distributor'],axis='columns',inplace=True)

# Normalize the dataset
scaler_trainx = MinMaxScaler()
scaler_trainy = MinMaxScaler()
scaler_testx = MinMaxScaler()
scaler_testy = MinMaxScaler()

target = df['Opening']
df = df.drop('Opening', axis=1)

X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=42)

X_train = scaler_trainx.fit_transform(pd.DataFrame(X_train))
y_train = scaler_trainy.fit_transform(pd.DataFrame(y_train))
X_test = scaler_testx.fit_transform(pd.DataFrame(X_test))
y_test = scaler_testy.fit_transform(pd.DataFrame(y_test))


def predict_revenue():
    theaters = int(theaters_entry.get())
    selected_distributor = distributor_var.get()
    
    data = pd.DataFrame({
        'Theaters': [theaters],
        '-': [0],
        '20th Century Studios': [0],
        'Columbia Pictures': [0],
        'Dimension Films': [0],
        'DreamWorks': [0],
        'DreamWorks Distribution': [0],
        'FilmDistrict': [0],
        'Focus Features': [0],
        'Lionsgate Films': [0],
        'Metro-Goldwyn-Mayer': [0],
        'Miramax': [0],
        'New Line Cinema': [0],
        'Newmarket Films': [0],
        'Paramount Pictures': [0],
        'Relativity Media': [0],
        'Revolution Studios': [0],
        'STX Entertainment': [0],
        'Screen Gems': [0],
        'Sony Pictures Releasing': [0],
        'Summit Entertainment': [0],
        'The Weinstein Company': [0],
        'TriStar Pictures': [0],
        'Twentieth Century Fox': [0],
        'United Artists Releasing': [0],
        'Universal Pictures': [0],
        'Universal Pictures International': [0],
        'Walt Disney Studios Motion Pictures': [0],
        'Warner Bros.': [0]
    })
    data[selected_distributor] = 1
    
    data_scaled = scaler_testx.transform(data)
    
    if model_var.get() == "Linear Regression":
        y_testing_scaled = model_lr.predict(data_scaled)
    elif model_var.get() == "Decision Tree Regression":
        y_testing_scaled = model_dt.predict(data_scaled)
    else:  # Random Forest Regression
        y_testing_scaled = model_rf.predict(data_scaled)
    
    # Reshape y_testing_scaled to 2D array
    y_testing_scaled = y_testing_scaled.reshape(-1, 1)
    
    y_testing = scaler_testy.inverse_transform(y_testing_scaled)
    average_predicted_revenue = y_testing.mean()
    
    result_label.config(text=f"Predicted Revenue: ${average_predicted_revenue:,.2f}")


def change_model(event):
    if model_var.get() == "Linear Regression":
        model_lr.fit(X_train, y_train)
    elif model_var.get() == "Decision Tree Regression":
        model_dt.fit(X_train, y_train)
    else:  # Random Forest Regression
        model_rf.fit(X_train, y_train.ravel())

# Initialize models
model_lr = LinearRegression()
model_dt = DecisionTreeRegressor(max_depth=100, min_samples_split=6, random_state=42)
model_rf = RandomForestRegressor(random_state=42)

root = tk.Tk() 
root.title("Movie Revenue Prediction")
root.geometry("800x600")


        
def exit_program():
    root.destroy()




bg_image = Image.open("machine5.jpg")
bg_image = bg_image.resize((1500, 800), Image.LANCZOS)
bg_image_tk = ImageTk.PhotoImage(bg_image)
canvas = tk.Canvas(root, width=1500, height=800)
canvas.grid(row=0, column=0, columnspan=4, sticky="nsew")
canvas.create_image(0, 0, image=bg_image_tk, anchor="nw")
canvas.bg_image_tk = bg_image_tk 


model_label = tk.Label(root, text="Select Model:")
model_label_canvas=canvas.create_window( 400, 10,  anchor = "nw", window = model_label)

model_var = tk.StringVar()
model_var.set("Linear Regression") 
model_dropdown = ttk.Combobox(root, textvariable=model_var, values=["Linear Regression", "Decision Tree Regression", "Random Forest Regression"], state='readonly')
model_dropdown_canvas=canvas.create_window( 600, 10,  anchor = "nw", window = model_dropdown)
model_dropdown.bind("<<ComboboxSelected>>", change_model)

theaters_label = tk.Label(root, text="Number of Theaters:")
theaters_label_canvas=canvas.create_window( 400, 60,  anchor = "nw", window = theaters_label)

theaters_entry = ttk.Entry(root)
theaters_entry_canvas=canvas.create_window( 600, 60,  anchor = "nw", window = theaters_entry)

distributor_label = tk.Label(root, text="Select Distributor:")
distributor_label_canvas=canvas.create_window( 400, 110,  anchor = "nw", window = distributor_label)

distributor_var = tk.StringVar()

distributor_dropdown = ttk.Combobox(root, textvariable=distributor_var, values=[
    '20th Century Studios', 'Columbia Pictures', 'Dimension Films', 'DreamWorks',
    'DreamWorks Distribution', 'FilmDistrict', 'Focus Features', 'Lionsgate Films',
    'Metro-Goldwyn-Mayer', 'Miramax', 'New Line Cinema', 'Newmarket Films',
    'Paramount Pictures', 'Relativity Media', 'Revolution Studios', 'STX Entertainment',
    'Screen Gems', 'Sony Pictures Releasing', 'Summit Entertainment', 'The Weinstein Company',
    'TriStar Pictures', 'Twentieth Century Fox', 'United Artists Releasing',
    'Universal Pictures', 'Universal Pictures International', 'Walt Disney Studios Motion Pictures',
    'Warner Bros.'
], state='readonly', width=50)  # Set the width and state
distributor_dropdown_canvas=canvas.create_window( 600, 110,  anchor = "nw", window =distributor_dropdown)




predict_button = tk.Button(root, text="Predict Revenue", command=predict_revenue , bg='darkorchid3', activebackground='darkorchid4')
predict_button_canvas=canvas.create_window( 500, 160,  anchor = "nw", window =predict_button)

result_label = tk.Label(root, text="")
result_label_canvas=canvas.create_window( 500, 210,  anchor = "nw", window =result_label)

exit_button = tk.Button(root, text="Exit", command=exit_program, bg='darkorchid3', activebackground='darkorchid4')
buttonexit_canvas = canvas.create_window( 1200, 600, anchor = "nw", window =exit_button) 


root.mainloop()


# In[2]:


pip install pyinstaller


# In[ ]:




