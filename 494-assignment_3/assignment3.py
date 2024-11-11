import opendatasets as od
import numpy as np 
import pandas as pd 
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os
sns.set_style('darkgrid')


dataset_url = 'https://www.kaggle.com/datasets/arjunbhasin2013/ccdata?resource=download'
od.download(dataset_url) # provide the kaggle API token here

path = './ccdata'
os.listdir(path)



train_csv = path + '/CC GENERAL.csv'
ccd = pd.read_csv(train_csv)



# prep the data 
# {
# remove the credit limit 1 null value
ccd[ccd['CREDIT_LIMIT'].isnull()]


# removed the row which contains the Nan Value
ccd.drop([5203], axis=0,inplace=True)


ccd.reset_index(inplace=True)
ccd.drop('index', axis=1, inplace=True)


# 313 null values
print(ccd[['PAYMENTS','MINIMUM_PAYMENTS']][(ccd['MINIMUM_PAYMENTS'].isnull())].shape[0])

# now there is another field that contains missing values which we shall input
# we can see we have 313 rows that are empty
# now we want to imput those rows
print(ccd[['PAYMENTS','MINIMUM_PAYMENTS']][(ccd['MINIMUM_PAYMENTS'].isnull()) & (ccd['PAYMENTS'] == 0)].shape[0])


payments_mean = np.mean(ccd['PAYMENTS'])
print(payments_mean)


# we have payments that are less than the mean, so it does make any sense to 
# impute the values of the empty columns[NaN columns] with the mean directly
# if the payments < payments_mean then use the values of the payments to impute -->
# --> the missing columns 

print(ccd[['PAYMENTS','MINIMUM_PAYMENTS']][(ccd['MINIMUM_PAYMENTS'].isnull()) & (ccd['PAYMENTS'] < payments_mean) &  (ccd['PAYMENTS'] > 0)].shape[0])


# is the payments > 0 and payments > mean
# then use the value of the mean to impute the missing value  
print(ccd[['PAYMENTS','MINIMUM_PAYMENTS']][(ccd['MINIMUM_PAYMENTS'].isnull()) & (ccd['PAYMENTS'] > payments_mean) & (ccd['PAYMENTS'] > 0)].shape[0])




# ccd.loc[(condition), 'column'] 
# 1st condition done
ccd.loc[(ccd['MINIMUM_PAYMENTS'].isnull()) & (ccd['PAYMENTS'] == 0), 'MINIMUM_PAYMENTS'] = 0

# 2nd condition done
ccd.loc[(ccd['MINIMUM_PAYMENTS'].isnull()) & (ccd['PAYMENTS'] < payments_mean) & (ccd['PAYMENTS'] > 0), 'MINIMUM_PAYMENTS'] = ccd['PAYMENTS']

# 3rd condition done
ccd.loc[(ccd['MINIMUM_PAYMENTS'].isnull()) & (ccd['PAYMENTS'] > payments_mean) & (ccd['PAYMENTS'] > 0), 'MINIMUM_PAYMENTS'] = payments_mean




# check the data 
ccd.info()

# we need to drop the ID's of the users
ccd = ccd.drop('CUST_ID',axis=1)

# } end of preping the data 

# Randomize the column selection


ccd_df = ccd.copy()
input_columns = list(ccd_df.columns[:-1])
y_column = list(ccd_df.columns[-1:])
selected_input_columns = random.sample(input_columns,k=int(len(input_columns)*0.80))


print(len(selected_input_columns))
print(len(input_columns))

print(y_column)



print(ccd_df[selected_input_columns])


X = ccd_df[selected_input_columns]
Y = ccd_df[y_column]


# display the selected columns 
print(selected_input_columns)



print(Y)


# KMeans Model 

from sklearn.cluster import KMeans
# automatically uses the euclidean algorithm
kmeans = KMeans(n_clusters=3, random_state=42)



kmeans.fit(X)
preds = kmeans.predict(X)

# the labels or the resulted clustering
print(np.unique(kmeans.labels_))


# due to random selection of columns this should change, every time the code is runned
wcss = kmeans.inertia_
print(wcss) #within-cluster sum of squares


### Plotting the Clusters
# inspired from this: https://www.kaggle.com/code/ahmadrafiee/customer-segmentation-eda-clustering-and-pca#-Step-3.1.4--%7C-Scatter-Plot


# designing how the plots will be
plt.subplots(nrows=3 , ncols=4 , figsize=(20,20))

#adjusting the plotted figures to be uniformly displayed
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.8)

palette1=["#ff7f0e","#2ca02c","#9467bd"]

j=0 
for i in selected_input_columns:   
    plt.subplot(3,4,j+1) # plotting the figures 1-by-1
    sns.scatterplot(x=i , y=selected_input_columns[0], hue=preds, data=X, palette=palette1)
    j=j+1









