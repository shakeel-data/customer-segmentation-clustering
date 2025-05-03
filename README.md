# 👤  Un-Supervised Customer Segmentation Project | KMeans Clustering
![image](https://github.com/user-attachments/assets/5468e320-c1f2-46c6-b84b-87a2c15fd688)

Customer segmentation is a critical practice for businesses as it helps them **divide their customer base into distinct groups based on shared characteristics, behaviors, or needs**. This approach allows businesses to tailor their marketing strategies, product offerings, and services more effectively. By understanding different customer segments, companies can target specific groups with personalized communication and promotions, increasing engagement, loyalty, and overall satisfaction. Additionally, segmentation helps in **optimizing resource allocation, improving sales conversion, and identifying new market opportunities, ultimately leading to higher profitability and sustainable growth.**

## 📘 Project Overview
This project focuses on customer segmentation using **unsupervised learning techniques to identify distinct groups within a customer base.** The goal is to help businesses make data-driven decisions such as personalized marketing, product recommendations, and resource allocation. The dataset used contains key features like **age, gender, annual income, and spending score.** These were preprocessed through steps including data **cleaning, encoding, and feature scaling.** The analysis aims to uncover hidden patterns in customer behavior that can be translated into **actionable business strategies.**

## 🎯 Key Objectives
**Data Exploration and Preprocessing:** Thoroughly **explore and clean the customer dataset** by handling missing values, normalizing features, and preparing the data for clustering.
**Segmentation of Customers:** Identify distinct customer segments based on **shared attributes and behaviors, providing a clear view** of the different customer groups.
**Clustering with K-Means:** Implement and fine-tune the K-Means clustering algorithm to **group customers into meaningful segments based on their characteristics.**
**Cluster Analysis and Insights:** Analyze the features of each customer segment to **uncover patterns and characteristics,** which can help design targeted marketing strategies and improve customer engagement.

## 📁 Data Sources
- Kaggle
  <a href="">csv</a>
- Python
  <a href="">codes</a>

## 🪜 Project Workflow
### 1. Load Packages and Data Ingestion

```python
#Importing the Libraries
import numpy as np
import pandas as pd
import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from sklearn import metrics
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
np.random.seed(42)
```

**Read the data**

```python
data = pd.read_csv("marketing_campaign.csv", sep="\t")
print("Number of datapoints:", len(data))
data.head()
```
![image](https://github.com/user-attachments/assets/9e1e29d9-565b-45c6-b7c2-15c64ca7d988)

## 2. 🧹 Data Cleaning

```python
data.info()
```
![image](https://github.com/user-attachments/assets/581287a9-bf81-4607-9ff6-babdfd4d746e)
**From the above output, we can conclude:**

- There are missing values in income.
- Dt_Customer is not parsed as DateTime.
- Some categorical features are in dtype: object, so we will need to encode them into numeric forms later.
- To handle the missing values, we will start by dropping the rows with missing income values.

```python
#To remove the NA values
data = data.dropna()
print("The total number of data-points after removing the rows with missing values are:", len(data))
```
**The total number of data-points after removing the rows with missing values are: 2216**

Next, we will create a feature from "Dt_Customer" to indicate the number of days a customer has been registered, relative to the most recent customer. To do this, we will first check the newest and oldest recorded dates.

## 3. Convert date with correct format - Dt_Customer

```python
data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"], dayfirst=True)

# Extract date only (if needed)
dates = data["Dt_Customer"].dt.date

# Print min and max dates
print("The newest customer's enrolment date in the records:", dates.max())
print("The oldest customer's enrolment date in the records:", dates.min())
```
**The newest customer's enrolment date in the records: 2014-06-29**
**The oldest customer's enrolment date in the records: 2012-07-30**

## 4. Created a feature Customer_For

```python
days = []
d1 = max(dates) #taking it to be the newest customer
for i in dates:
    delta = d1 - i
    days.append(delta)
data["Customer_For"] = days
data["Customer_For"] = pd.to_numeric(data["Customer_For"], errors="coerce")
```
```python
print("Total categories in the feature Marital_Status:\n", data["Marital_Status"].value_counts(), "\n")
print("Total categories in the feature Education:\n", data["Education"].value_counts())
```
![image](https://github.com/user-attachments/assets/1c7bc0d3-8051-4150-bdae-e1dda78db866)

- Extract "Age" from "Year_Birth" to indicate the customer's age.
- Create a "Spent" feature representing the total amount spent by the customer over two years.
- Derive "Living_With" from "Marital_Status" to reflect the living situation of couples.
- Create a "Children" feature to indicate the total number of children (kids and teenagers) in the household.
- Add a "Family_Size" feature for better household clarity.
- Create "Is_Parent" to indicate whether the customer is a parent.
- Simplify "Education" into three categories.
- Drop redundant features

## 5. ⚙️ Feature Engineering

```python
#Age of customer today 
data["Age"] = 2021-data["Year_Birth"]

#Total spendings on various items
data["Spent"] = data["MntWines"]+ data["MntFruits"]+ data["MntMeatProducts"]+ data["MntFishProducts"]+ data["MntSweetProducts"]+ data["MntGoldProds"]

#Deriving living situation by marital status"Alone"
data["Living_With"]=data["Marital_Status"].replace({"Married":"Partner", "Together":"Partner", "Absurd":"Alone", "Widow":"Alone", "YOLO":"Alone", "Divorced":"Alone", "Single":"Alone",})

#Feature indicating total children living in the household
data["Children"]=data["Kidhome"]+data["Teenhome"]

#Feature for total members in the householde
data["Family_Size"] = data["Living_With"].replace({"Alone": 1, "Partner":2})+ data["Children"]

#Feature pertaining parenthood
data["Is_Parent"] = np.where(data.Children> 0, 1, 0)

#Segmenting education levels in three groups
data["Education"]=data["Education"].replace({"Basic":"Undergraduate","2n Cycle":"Undergraduate", "Graduation":"Graduate", "Master":"Postgraduate", "PhD":"Postgraduate"})

#For clarity
data=data.rename(columns={"MntWines": "Wines","MntFruits":"Fruits","MntMeatProducts":"Meat","MntFishProducts":"Fish","MntSweetProducts":"Sweets","MntGoldProds":"Gold"})

#Dropping some of the redundant features
to_drop = ["Marital_Status", "Dt_Customer", "Z_CostContact", "Z_Revenue", "Year_Birth", "ID"]
data = data.drop(to_drop, axis=1)
```

```python
data.describe()
```
**8 rows × 28 columns**

**The discrepancies in mean and max values for Income and Age are noted, with the max age being 128 due to the data being from an older source. We will now visualize some selected features for a broader view of the data.**

```python
#To plot some selected features 
#Setting up colors prefrences
sns.set(rc={"axes.facecolor":"#FFF9ED","figure.facecolor":"#FFF9ED"})
pallet = ["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"]
cmap = colors.ListedColormap(["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"])
#Plotting following features
To_Plot = [ "Income", "Recency", "Customer_For", "Age", "Spent", "Is_Parent"]
print("Reletive Plot Of Some Selected Features: A Data Subset")
plt.figure()
sns.pairplot(data[To_Plot], hue= "Is_Parent",palette= (["#682F2F","#F3AB60"]))
#Taking hue 
plt.show()
```
![image](https://github.com/user-attachments/assets/11a42845-3de3-4226-b239-5c8fdc553afb)

```python
#Dropping the outliers by setting a cap on Age and income. 
data = data[(data["Age"]<90)]
data = data[(data["Income"]<600000)]
print("The total number of data-points after removing the outliers are:", len(data))
```
**The total number of data-points after removing the outliers are: 2212**

## 6. 🌡️ Correlation Matrix
```python
plt.figure(figsize=(25, 25))
plt.title('Customer Segmentation Features Correlation Plot')

# Use the correct DataFrame and ensure only numeric columns are included
corr = data.select_dtypes(include='number').corr()
sns.heatmap(corrmat,annot=True, cmap=cmap, center=0)
```
![image](https://github.com/user-attachments/assets/a0c3e6f3-a041-43d5-a8f0-a0fc540f44b4)

## 7. 🔁 Data Preprosessing
we will prepare the data for clustering by applying the following preprocessing steps:
**Steps:**
a. Encode categorical features using labels
b. Standardize the features with StandardScaler
c. Generate a subset dataframe for dimensionality reduction

```python
#Get list of categorical variables
s = (data.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables in the dataset:", object_cols)
```
**Categorical variables in the dataset: ['Education', 'Living_With']**
```python
#Label Encoding the object dtypes.
LE=LabelEncoder()
for i in object_cols:
    data[i]=data[[i]].apply(LE.fit_transform)
print("All features are now numerical")
```
**All features are now numerical**

```python
#Creating a copy of data
ds = data.copy()
# creating a subset of dataframe by dropping the features on deals accepted and promotions
cols_del = ['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1','AcceptedCmp2', 'Complain', 'Response']
ds = ds.drop(cols_del, axis=1)
#Scaling
scaler = StandardScaler()
scaler.fit(ds)
scaled_ds = pd.DataFrame(scaler.transform(ds),columns= ds.columns )
print("All features are now scaled")
```
**All features are now scaled**

```python
#Scaled data to be used for reducing the dimensionality
print("Dataframe to be used for further modelling:")
scaled_ds.head()
```
**
| Education  | Income    | Kidhome   | Teenhome  | Recency   | Wines     | Fruits    | Meat      | Fish      | Sweets    | ... | NumCatalogPurchases | NumStorePurchases | NumWebVisitsMonth | Customer_For | Age       | Spent     | Living_With | Children  | Family_Size | Is_Parent |
|------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----|----------------------|--------------------|--------------------|---------------|-----------|-----------|--------------|-----------|--------------|------------|
| -0.893586  | 0.287105  | -0.822754 | -0.929699 | 0.310353  | 0.977660  | 1.552041  | 1.690293  | 2.453472  | 1.483713  | ... | 2.503607             | -0.555814          | 0.692181           | 1.527721      | 1.018352  | 1.676245  | -1.349603    | -1.264598 | -1.758359    | -1.581139  |
| -0.893586  | -0.260882 | 1.040021  | 0.908097  | -0.380813 | -0.872618 | -0.637461 | -0.718230 | -0.651004 | -0.634019 | ... | -0.571340            | -1.171160          | -0.132545          | -1.189011     | 1.274785  | -0.963297 | -1.349603    | 1.404572  | 0.449070     | 0.632456   |
| -0.893586  | 0.913196  | -0.822754 | -0.929699 | -0.795514 | 0.357935  | 0.570540  | -0.178542 | 1.339513  | -0.147184 | ... | -0.229679            | 1.290224           | -0.544908          | -0.206048     | 0.334530  | 0.280110  | 0.740959     | -1.264598 | -0.654644    | -1.581139  |
| -0.893586  | -1.176114 | 1.040021  | -0.929699 | -0.795514 | -0.872618 | -0.561961 | -0.655787 | -0.504911 | -0.585335 | ... | -0.913000            | -0.555814          | 0.279818           | -1.060584     | -1.289547 | -0.920135 | 0.740959     | 0.069987  | 0.449070     | 0.632456   |
| 0.571657   | 0.294307  | 1.040021  | -0.929699 | 1.554453  | -0.392257 | 0.419540  | -0.218684 | 0.152508  | -0.001133 | ... | 0.111982             | 0.059532           | -0.132545          | -0.951915     | -1.033114 | -0.307562 | 0.740959     | 0.069987  | 0.449070     | 0.632456   |
5 rows × 23 columns





























































































































