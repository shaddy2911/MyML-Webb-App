from pandas.core.base import PandasObject
from pandas.io.parsers import read_csv
import streamlit as st 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import pandas as pd
#from sklearn import datasets
from sklearn.model_selection import train_test_split
#from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from PIL import Image
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

#set title
st.title('MyML App')
image=Image.open('1pic.jpg')
st.image(image,use_column_width=True)


def main():
    activites=['EDA','Visualization','Model','About Us']
    option=st.sidebar.selectbox('Select The Fungtion You Want:',activites)
    if option=='EDA':
        st.subheader('Exploratory Data Analysis')
        data=st.file_uploader('Upload The Dataset',type=['csv'])
        st.success('Data Successfully Uploaded')
        if data is not None:
            df=pd.read_csv(data)
            st.dataframe(df.head(20))
            if st.checkbox('Display Shape'):
                st.write(df.shape)
            if st.checkbox('Display Columns'):
                st.write(df.columns)
            if st.checkbox('Select Multiple Columns'):
                selected_columns=st.multiselect('Select Preferred Columns',df.columns)
                df1=df[selected_columns]
                st.dataframe(df1)
            if st.checkbox('Display Summary'):
                st.write(df.describe().T)
            if st.checkbox('Display Null Values'):
                st.write(df.isnull().sum())
            if st.checkbox('Display The Data Type'):
                st.write(df.dtypes)
            if st.checkbox('Display Correlation'):
                st.write(df.corr())
    

    if option=='Visualization':
        st.subheader('Exploratory Data Analysis')
        data=st.file_uploader('Upload The Dataset',type=['csv'])
        st.success('Data Successfully Uploaded')
        if data is not None:
            df=pd.read_csv(data)
            st.dataframe(df.head(20))
            if st.checkbox('Select Columns To Plot'):
                selected_columns=st.multiselect('Select Your Preferred Columns',df.columns)
                df1=df[selected_columns]
                st.dataframe(df1)
            if st.checkbox('Display Heatmap'):
                st.write(sns.heatmap(df1.corr(),vmax=1,square=True,cmap='viridis'))
                st.pyplot()
                st.set_option('deprecation.showPyplotGlobalUse', False)
            if st.checkbox('Display Pairplot'):
                st.write(sns.pairplot(df1,diag_kind='kde'))
                st.pyplot()
            if st.checkbox('Display Pie-Chart'):
                all_columns=df.columns.to_list()
                pie_columns=st.selectbox('Select Columns To Display',all_columns)
                pie_chart=df[pie_columns].value_counts().plot.pie(autopct='%1.1f%%')
                st.write(pie_chart)
                st.pyplot()

    
    if option=='Model':
        st.subheader('Model Building')
        data=st.file_uploader('Upload The Dataset',type=['csv'])
        st.success('Data Successfully Uploaded')
        if data is not None:
            df=pd.read_csv(data)
            st.dataframe(df.head(20))

            if st.checkbox('Select Multiple Columns'):
                new_data=st.multiselect('Select Your Preferred Columns',df.columns)
                df1=df[new_data]
                st.dataframe(df1)

                #dividig data into x and y columns

                x=df1.iloc[:,0:-1]
                y=df1.iloc[:,-1]
            seed=st.sidebar.slider('Random States',1,200)
            classifier_name=st.sidebar.selectbox('Select Your Preferred Classifier',('KNN','LR'))

            def add_parameter(name_of_clf):
                param=dict()
                
                if name_of_clf=='KNN':
                    K=st.sidebar.slider('K',1,15)
                    param['K']=K
                    return param
            
            param=add_parameter(classifier_name)

            #defining function for classifier

            def get_classifier(name_of_clf,param):
                clf= None
                if name_of_clf=='KNN':
                    clf=KNeighborsClassifier(n_neighbors=param['K'])
                
                elif name_of_clf=='LR':
                    clf=LogisticRegression()
                
                else:
                    st.warning('Select Your Preferred Algorithm')
                return clf

            clf=get_classifier(classifier_name,param)

            x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=seed)
            clf.fit(x_train,y_train)

            y_pred=clf.predict(x_test)
            st.write('Predictions:',y_pred)
            accuracy=accuracy_score(y_test,y_pred)
            st.write('Name Of Classifier:',classifier_name)
            st.write('Accuracy is:',accuracy)
            
        
if __name__ == '__main__':
    main()
