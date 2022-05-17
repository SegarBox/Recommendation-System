import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn.decomposition import TruncatedSVD

# %matplotlib inline
plt.style.use("ggplot")

amazon_ratings = pd.read_csv(r'D:\13_Bangkit Academy 2022\Capstone Project\Dataset\Amazon - Ratings (Beauty Products)\ratings_Beauty.csv')
print(amazon_ratings)
amazon_ratings = amazon_ratings.dropna()
amazon_ratings.head()

popular_products = pd.DataFrame(amazon_ratings.groupby('ProductId')['Rating'].count())
most_popular = popular_products.sort_values('Rating', ascending=False)
most_popular.head(10)

most_popular.head(10).plot(kind="bar")

# Subset of Amazon Ratings
amazon_ratings1 = amazon_ratings.head(10000)

ratings_utility_matrix = amazon_ratings1.pivot_table(values='Rating', index='UserId', columns='ProductId', fill_value=0)
ratings_utility_matrix.head()

X = ratings_utility_matrix.T
X.head()

SVD = TruncatedSVD(n_components=10)
decomposed_matrix = SVD.fit_transform(X)
correlation_matrix = np.corrcoef(decomposed_matrix)

num = int(input("Enter number :"))
print(num)

# Printing type of input value 
print("type of number", type(num))

i = X.index[num]

product_names = list(X.index)
product_ID = product_names.index(i)

correlation_product_ID = correlation_matrix[product_ID]

Recommend: list[Any] = list(X.index[correlation_product_ID > 0.90])

# Removes the item already bought by the customer
Recommend.remove(i)
Recommend[0:9]
