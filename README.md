# 👥  Un-Supervised Customer Segmentation Project | KMeans Clustering
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













































































































































