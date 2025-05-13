# 👤  Un-Supervised Customer Segmentation Project | KMeans Clustering kffffnfjenfjwb kxsnxjsj kdw djwnj wmdwmkm wmri3jrim ynuuk dsdnjn dwkdk ekndjn
![image](https://github.com/user-attachments/assets/5468e320-c1f2-46c6-b84b-87a2c15fd688)

Customer segmentation is a critical practice for businesses as it helps them **divide their customer base into distinct groups based on shared characteristics, behaviors, or needs**. This approach allows businesses to tailor their marketing strategies, product offerings, and services more effectively. By understanding different customer segments, companies can target specific groups with personalized communication and promotions, increasing engagement, loyalty, and overall satisfaction. Additionally, segmentation helps in **optimizing resource allocation, improving sales conversion, and identifying new market opportunities, ultimately leading to higher profitability and sustainable growth.**

## 📘 Project Overview
This project focuses on customer segmentation using **unsupervised learning techniques to identify distinct groups within a customer base.** The goal is to help businesses make data-driven decisions such as personalized marketing, product recommendations, and resource allocation. The dataset used contains key features like **age, gender, annual income, and spending score.** These were preprocessed through steps including data **cleaning, encoding, and feature scaling.** The analysis aims to uncover hidden patterns in customer behavior that can be translated into **actionable business strategies.**

## 🎯 Key Objectives
- **Data Exploration and Preprocessing:** Thoroughly **explore and clean the customer dataset** by handling missing values, normalizing features, and preparing the data for clustering.
- **Segmentation of Customers:** Identify distinct customer segments based on **shared attributes and behaviors, providing a clear view** of the different customer groups.
- **Clustering with K-Means:** Implement and fine-tune the K-Means clustering algorithm to **group customers into meaningful segments based on their characteristics.**
- **Cluster Analysis and Insights:** Analyze the features of each customer segment to **uncover patterns and characteristics,** which can help design targeted marketing strategies and improve customer engagement.

## 📁 Data Sources
- Kaggle
  <a href="https://github.com/shakeel-data/customer-segmentation-clustering/blob/main/marketing_campaign.csv">csv</a>
- Python
  <a href="https://github.com/shakeel-data/customer-segmentation-clustering/blob/main/customer_segmentation_clustering.ipynb">codes</a>

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

## 3. Convert date with correct format – Dt_Customer

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

## 4. Create a feature – Customer_For

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
| Metric | Income     | Kidhome | Teenhome | Recency | Wines     | Fruits   | Meat      | Fish     | Sweets   | Age     | Spent    | Family_Size  | Is_Parent  |
|--------|------------|---------|----------|---------|-----------|----------|-----------|----------|----------|---------|----------|--------------|------------|
| Count  | 2216       | 2216    | 2216     | 2216    | 2216      | 2216     | 2216      | 2216     | 2216     | 2216    | 2216     | 2216         | 2216       |
| Mean   | 52247.25   | 0.44    | 0.51     | 49.01   | 305.09    | 26.36    | 166.99    | 37.64    | 27.03    | 52.18   | 607.08   | 2.59         | 0.71       |
| Std    | 25173.08   | 0.54    | 0.54     | 28.95   | 337.33    | 39.79    | 224.28    | 54.75    | 41.07    | 11.99   | 602.90   | 0.91         | 0.45       |
| Min    | 1730       | 0       | 0        | 0       | 0         | 0        | 0         | 0        | 0        | 25      | 5        | 1            | 0          |
| 25%    | 35303      | 0       | 0        | 24      | 24        | 2        | 16        | 3        | 1        | 44      | 69       | 2            | 0          |
| 50%    | 51381.5    | 0       | 0        | 49      | 174.5     | 8        | 68        | 12       | 8        | 51      | 396.5    | 3            | 1          |
| 75%    | 68522      | 1       | 1        | 74      | 505       | 33       | 232.25    | 50       | 33       | 62      | 1048     | 3            | 1          |
| Max    | 666666     | 2       | 2        | 99      | 1493      | 199      | 1725      | 259      | 262      | 128     | 2525     | 5            | 1          |

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

**5 rows × 23 columns**

## 8. 🧬 Dimensionality Reduction
We will perform dimensionality reduction using Principal Component Analysis (PCA) to reduce correlated and redundant features before classification. PCA minimizes information loss while increasing interpretability. The dimensions will be reduced to 3.

**Steps:**
- Apply PCA for dimensionality reduction
- Plot the reduced dataframe

```python
#Initiating PCA to reduce dimentions aka features to 3
pca = PCA(n_components=3)
pca.fit(scaled_ds)
PCA_ds = pd.DataFrame(pca.transform(scaled_ds), columns=(["col1","col2", "col3"]))
PCA_ds.describe().T
```

| Metric | col1       | col2       | col3       |
|--------|------------|------------|------------|
| Count  | 2212.0     | 2212.0     | 2212.0     |
| Mean   | 0.000000   | 0.000000   | -0.000000  |
| Std    | 2.878602   | 1.709469   | 1.231687   |
| Min    | -5.978124  | -4.194757  | -3.625248  |
| 25%    | -2.539470  | -1.323929  | -0.853713  |
| 50%    | -0.781595  | -0.173721  | -0.050842  |
| 75%    | 2.386380   | 1.234851   | 0.863974   |
| Max    | 7.452915   | 6.168189   | 6.750458   |

### 🧊 3D reduced Dimension

```python
#A 3D Projection Of Data In The Reduced Dimension
x =PCA_ds["col1"]
y =PCA_ds["col2"]
z =PCA_ds["col3"]
#To plot
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x,y,z, c="maroon", marker="o" )
ax.set_title("A 3D Projection Of Data In The Reduced Dimension")
plt.show()
```
![image](https://github.com/user-attachments/assets/2403165c-ddd6-4982-a064-bd047496af5e)

## 9. 🔴 Clustering
We will apply Agglomerative Clustering, a hierarchical method that iteratively merges data points until the target number of clusters is formed, following dimensionality reduction to three features.

### 🔺 Elbow Method
```python
# Quick examination of elbow method to find numbers of clusters to make.
print('Elbow Method to determine the number of clusters to be formed:')
Elbow_M = KElbowVisualizer(KMeans(), k=10)
Elbow_M.fit(PCA_ds)
Elbow_M.show()
```
![image](https://github.com/user-attachments/assets/e0aa0800-5910-4e16-ae22-7b4e19fd16ba)

### 🌿 Agglomerative Clustering Model
```python
#Initiating the Agglomerative Clustering model 
AC = AgglomerativeClustering(n_clusters=4)
# fit model and predict clusters
yhat_AC = AC.fit_predict(PCA_ds)
PCA_ds["Clusters"] = yhat_AC
#Adding the Clusters feature to the orignal dataframe.
data["Clusters"]= yhat_AC
```

### 🧊 3D distribution
```python
#Plotting the clusters
fig = plt.figure(figsize=(10,8))
ax = plt.subplot(111, projection='3d', label="bla")
ax.scatter(x, y, z, s=40, c=PCA_ds["Clusters"], marker='o', cmap = cmap )
ax.set_title("The Plot Of The Clusters")
plt.show()
```
![image](https://github.com/user-attachments/assets/ca3f43b8-7853-4b7d-9862-5b3a659aa29a)

## 10. 🤖 Evaluating Models
As this is an unsupervised clustering task, there is no labeled target for evaluation. Instead, we analyze the clusters through exploratory data analysis to uncover meaningful patterns. We begin by reviewing the distribution of data points across the clusters.

```python
#Plotting countplot of clusters
pal = ["#682F2F","#B9C0C9", "#9F8A78","#F3AB60"]
pl = sns.countplot(x=data["Clusters"], palette= pal)
pl.set_title("Distribution Of The Clusters")
plt.show()
```
![image](https://github.com/user-attachments/assets/84de830b-299e-41d9-b6b9-447e71a16cdc)

```python
pl = sns.scatterplot(data = data,x=data["Spent"], y=data["Income"],hue=data["Clusters"], palette= pal)
pl.set_title("Cluster's Profile Based On Income And Spending")
plt.legend()
plt.show()
```
![image](https://github.com/user-attachments/assets/db9dfd4d-6643-4f68-a4d4-61710052c49a)
**The Income vs. Spending plot reveals distinct cluster patterns:**
- Group 0: High spending, average income
- Group 1: High spending, high income
- Group 2: Low spending, low income
- Group 3: High spending, low income

```python
plt.figure()
pl=sns.swarmplot(x=data["Clusters"], y=data["Spent"], color= "#CBEDDD", alpha=0.5 )
pl=sns.boxenplot(x=data["Clusters"], y=data["Spent"], palette=pal)
plt.show()
```
![image](https://github.com/user-attachments/assets/351df264-94eb-4054-80d0-438290d04c6a)
- From the plot, it's clear that Cluster 1 represents the largest group of customers, followed closely by Cluster 0.
- We can now explore the spending behavior within each cluster to help inform targeted marketing strategies.

```python
#Creating a feature to get a sum of accepted promotions 
data["Total_Promos"] = data["AcceptedCmp1"]+ data["AcceptedCmp2"]+ data["AcceptedCmp3"]+ data["AcceptedCmp4"]+ data["AcceptedCmp5"]
#Plotting count of total campaign accepted.
plt.figure()
pl = sns.countplot(x=data["Total_Promos"],hue=data["Clusters"], palette= pal)
pl.set_title("Count Of Promotion Accepted")
pl.set_xlabel("Number Of Total Accepted Promotions")
plt.show()
```
![image](https://github.com/user-attachments/assets/e294a370-930a-4cee-b509-5cdbc6fb745a)
- The response to past campaigns has been underwhelming, with very few participants overall and none engaging in all five campaigns.
- This suggests a need for better-targeted and more strategically planned campaigns to effectively boost sales.

```python
#Plotting the number of deals purchased
plt.figure()
pl=sns.boxenplot(y=data["NumDealsPurchases"],x=data["Clusters"], palette= pal)
pl.set_title("Number of Deals Purchased")
plt.show()
```
![image](https://github.com/user-attachments/assets/ccd956b4-463c-4449-a541-fbd0dd8c8ff4)
- Unlike the campaigns, the deals performed well—especially with Cluster 0 and Cluster 3.
- However, our high-value Cluster 1 customers show little interest in deals, and Cluster 2 remains largely unresponsive to both campaigns and deals.

```python
#for more details on the purchasing style 
Places =["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases",  "NumWebVisitsMonth"] 

for i in Places:
    plt.figure()
    sns.jointplot(x=data[i],y = data["Spent"],hue=data["Clusters"], palette= pal)
    plt.show()
```
![image](https://github.com/user-attachments/assets/96cdf1ea-075e-422d-ab27-9eb0edc92707)
![image](https://github.com/user-attachments/assets/7984e741-c668-4f36-b159-3685c1d8cede)
![image](https://github.com/user-attachments/assets/0591a0cb-cb54-42d8-a263-1ca8117c39ba)
![image](https://github.com/user-attachments/assets/528d62bc-c95f-4d55-b00b-7ee6c820c4c2)

## 11. 🧾 Profiling
With clusters formed and purchasing habits analyzed, we'll now profile customers by plotting key personal traits across clusters. This will help identify star customers and those needing more marketing attention
![image](https://github.com/user-attachments/assets/a8d2a338-9661-4d8b-a3e5-fd7d434a2005)

```python
Personal = [ "Kidhome","Teenhome","Customer_For", "Age", "Children", "Family_Size", "Is_Parent", "Education","Living_With"]

for i in Personal:
    plt.figure()
    sns.jointplot(x=data[i], y=data["Spent"], hue =data["Clusters"], kind="kde", palette=pal)
    plt.show()
```
![image](https://github.com/user-attachments/assets/d4359378-c12e-47b1-9626-e96b739c98e9)
![image](https://github.com/user-attachments/assets/5a547cde-d0c7-47c6-9b5c-48fba080f1d2)
![image](https://github.com/user-attachments/assets/df1eb985-8acd-4ecc-add0-c28764e77971)
![image](https://github.com/user-attachments/assets/9a33befd-0b63-486f-9dc9-669bd10f27b1)
![image](https://github.com/user-attachments/assets/27c50c71-869b-4e12-8f27-5097ef4afea7)
![image](https://github.com/user-attachments/assets/81301a28-5dfc-4672-a018-f0babff2d76e)
![image](https://github.com/user-attachments/assets/e79dbfaf-3422-4210-a8b1-a63af15cc6d8)
![image](https://github.com/user-attachments/assets/f749f21b-ad6e-430d-aff9-da7854c55c6c)
![image](https://github.com/user-attachments/assets/7048d3db-7c79-4b7a-9b8d-4a99abdf894c)



## 🌟 Highlights and Key Insights
- **Cluster Discovery with K-Means:** The analysis successfully segmented customers into **4 distinct groups using the K-Means clustering algorithm,** enabling a structured understanding of different customer behaviors and profiles.
- **Optimal Cluster Validation:** To ensure the accuracy and relevance of the clustering, techniques like the **Elbow Method and Silhouette Score were utilized.** These methods confirmed that dividing the dataset into four clusters provided the best balance between **compactness and separation.**
- **Dimensionality Reduction for Visualization:** Principal Component Analysis **(PCA)** was applied to reduce the high-dimensional feature space into two principal components, allowing for clear and insightful **3D visualizations of the clusters.**
- **Insightful Cluster Profiling:** Each cluster was analyzed based on key features such as **income, spending score, age, family size, and purchasing behavior.** This revealed meaningful segments like high-income low spenders, young high spenders, and other behavior-based groups, offering actionable insights for **personalized marketing strategies.**

## ☁️ Technologies and Tools
- **Kaggle** – Dataset source
- **Jupyter Notebook** – Interactive environment for coding and presenting analysis
- **Python**
  - Libraries: ```numpy```, ```pandas```, ```matplotlib```, ```seaborn```
- **Machine Learning** – Model development and evaluation
  - Scikit-learn: ```KMeans```, ```PCA ```, ```StandardScaler ```, ```LabelEncoder ```, ```AgglomerativeClustering ```, ```Metrics ```
  - Yellowbrick: ```KElbowVisualizer```
  - mpl_toolkits (Axes3D): ```mplot3d```
  - Metrics: ```silhouette_score```
  - Warnings & Sys: Handling runtime warnings and system-level settings

## ✅ Conclusion
The K-Means clustering analysis successfully identified distinct customer segments based on key characteristics and behaviors, such as income, spending patterns, and demographic information. These segments provide actionable insights that can be leveraged to design targeted marketing campaigns, personalized offers, and loyalty programs that better align with customer needs.

While the analysis offers valuable insights, it does have some limitations. The dataset used is relatively small and lacks dynamic behavioral data over time, which could affect the robustness of the segmentation. Additionally, the model's performance could improve with more detailed and comprehensive data.

