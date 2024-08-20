import pandas as pd
from matplotlib import pyplot as plt

movie_data = pd.read_csv("movie_data/movies.dat", delimiter="::", encoding="windows-1252")
ratings = pd.read_csv("movie_data/ratings.dat", delimiter="::", usecols=["UserID", "MovieID", "Rating"])
user_data = pd.read_csv("movie_data/users.csv", usecols=["UserID", "Gender", "Occupation", "Age"])

# plot_data = {}
# age_group_counts = {}
# for i, elem in enumerate(ratings.UserID):
#     try:
#         age_group_counts[user_data["Age"][elem - 1]] += 1
#     except KeyError:
#         age_group_counts[user_data["Age"][elem - 1]] = 1
# print(age_group_counts)
#
# movie_id = 1
# for i, elem in enumerate(ratings.MovieID):
#     if elem == movie_id:
#         user_id = ratings["UserID"][i]
#         try:
#             plot_data[user_data["Age"][user_id - 1]] += 1
#         except KeyError:
#             plot_data[user_data["Age"][user_id - 1]] = 1
#
# print(plot_data)
# plot_data = {key: plot_data[key]/age_group_counts[key] for key in plot_data}
# print(plot_data)

# movie_id = 1570
# review_table = {}

# for i, elem in enumerate(ratings.MovieID):
#     if elem == movie_id:
#         try:
#             review_table[ratings["Rating"][i]] += 1
#         except KeyError:
#             review_table[ratings["Rating"][i]] = 1
#
# print(review_table)
#
# plt.bar(x=review_table.keys(), height=review_table.values())
# plt.show()

# def common_elems(l1, l2):
#     c = 0
#     for elem in l1:
#         if elem in l2:
#             c += 1
#     return c
#
# def movie_from_id(id):
#     for i, elem in enumerate(movie_data.MovieID):
#         if elem == id:
#             return movie_data.Movie[i]
#
# def top_genre(genres):
#     genre_table = {}
#     for i, elem in enumerate(movie_data["Genre"]):
#         genre_table[movie_data["MovieID"][i]] = common_elems(elem.split("|"), genres)
#     movie_table = {}
#     for i, elem in enumerate(ratings["MovieID"]):
#         try:
#             movie_table[elem] += genre_table[elem]
#         except KeyError:
#             movie_table[elem] = genre_table[elem]
#     x = lambda k: movie_table[k]
#     sorted_keys = sorted(movie_table.keys(), key=x, reverse=True)
#     movie_table = {key: movie_table[key] for key in sorted_keys if movie_table[key] != 0}
#     return movie_table
#
#
# movies = top_genre(["Action", "Crime", "Thriller"])
# movies = {movie_from_id(k): movies[k] for k in movies}
# for k in movies:
#     print(f"{k} : {movies[k]}")