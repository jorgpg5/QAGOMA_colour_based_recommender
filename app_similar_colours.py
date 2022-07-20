import os
import cv2
import random
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt

from PIL import Image
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
from webcolors import (CSS3_HEX_TO_NAMES, hex_to_rgb, )

st.set_page_config(layout="wide")

N_COLOURS = 5
IMG_SIZE = 224
IMG_DATASET = 'F:/QAGOMA_AI_technologist/img_dataset/'
UPLOADED_IMAGES_FOLDER = 'F:/QAGOMA_AI_technologist/recommendation_engine/colour_clustering/streamlit_prototype/images/'

header_col1, header_col2, header_col3 = st.columns(3)
with header_col3:
	logo_img = Image.open('qagoma_logo.png')
	st.image(logo_img)

st.title('QAGOMA - Colour Similarity Recommendation System Prototype')
file = st.file_uploader("Upload an artwork", type=['jpg', 'png', 'jpeg'])
layout_col1, layout_col2, layout_col3, layout_col4 = st.columns(4)


# Aux functions ###


def save_uploaded_file(uploaded_file):
	with open(os.path.join("images", uploaded_file.name), "wb") as f:
		f.write(uploaded_file.getbuffer())


def convert_rgb_to_names(rgb_tuple):
	# a dictionary of all the hex and their respective names in css3
	css3_db = CSS3_HEX_TO_NAMES  # 147 colours
	names = []
	rgb_values = []
	for color_hex, color_name in css3_db.items():
		names.append(color_name)
		rgb_values.append(hex_to_rgb(color_hex))

	kdt_db = KDTree(rgb_values)
	distance, index = kdt_db.query(rgb_tuple)
	return names[index]


def extracting_colour_names_from_hex(hex_vals):
	CSS_colours = []
	for value in hex_vals:
		if type(value) != str:
			CSS_colours.append('')

		else:
			rgb_value = hex_to_rgb(value)
			CSS_colours.append(convert_rgb_to_names(rgb_value))

	return CSS_colours


def visualise_colour_recommendations(selected_artworks_df):
	for item, title in zip(selected_artworks_df['Image Filename'], selected_artworks_df.Title):
		filename = os.path.join(IMG_DATASET, item)
		img = load_single_img(filename)
		img = np.array(img).reshape(IMG_SIZE, IMG_SIZE, 3)
		print('An artwork with similar colours may be "{}"'.format(title))

		plt.imshow(img)
		plt.axis('off')
		plt.show()


def load_single_img(filename):
	x_img = cv2.imread(filename, 1).astype('uint8')
	x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)
	x_img = cv2.resize(x_img, (IMG_SIZE, IMG_SIZE))
	x_height, x_width, channels = x_img.shape
	x_img = np.array(x_img).reshape(1, x_height, x_width, 3)

	return x_img


def get_artworks_with_similar_colours(df, placeholder):
	if len(df) >= 5:
		selected_artworks_df = df.loc[random.sample(list(df.index), 5)]
		print(selected_artworks_df)
	cols_img_list = []
	cols_caption_list = []
	for item, title in zip(selected_artworks_df['Image Filename'], selected_artworks_df.Title):
		filename = os.path.join(IMG_DATASET, item)
		img = Image.open(filename)
		cols_caption_list.append('"{}"'.format(title))
		cols_img_list.append(img)

	with placeholder.container():

		st.header('Artworks with similar colours')
		col1, col2, col3, col4, col5 = st.columns(5)
		col1.image(cols_img_list[0], use_column_width=True)
		col1.caption(cols_caption_list[0])
		col2.image(cols_img_list[1], use_column_width=True)
		col2.caption(cols_caption_list[1])
		col3.image(cols_img_list[2], use_column_width=True)
		col3.caption(cols_caption_list[2])
		col4.image(cols_img_list[3], use_column_width=True)
		col4.caption(cols_caption_list[3])
		col5.image(cols_img_list[4], use_column_width=True)
		col5.caption(cols_caption_list[4])


def refresh_button(placeholder):
	get_artworks_with_similar_colours(colour_matches_df, placeholder)


def main():
	global colour_matches_df
	placeholder = st.empty()
	dataset_df = pd.read_csv('image_dataset_may_2022_CSS_colours.csv')

	if file is not None:
		img = Image.open(file)
		# print(file.name)
		save_uploaded_file(file)

		pixels = np.array(list(img.getdata()))

		# fit KMeans and get centroids
		kmeans = KMeans(n_clusters=N_COLOURS)
		results = kmeans.fit(pixels)

		centroids = results.cluster_centers_
		labels = list(kmeans.labels_)

		percent = []
		for i in range(len(centroids)):
			j = labels.count(i)
			j = j / (len(labels))
			percent.append(j)

		int_centroids = []
		for item in centroids:
			int_items = [int(x) for x in item]
			int_centroids.append(int_items)

		with layout_col2:
			st.image(img, caption='This is your uploaded image')
		with layout_col3:
			fig1, ax1 = plt.subplots()
			ax1.pie(percent, colors=np.array(centroids/255),labels=np.arange(len(centroids)))
			st.pyplot(fig1)
		int_centroids = np.array(int_centroids)

		hex_colours_img = []
		for rgb_colour in int_centroids:
			hex_colours_img.append(matplotlib.colors.to_hex(rgb_colour / 255, keep_alpha=False))

		CSS_colours_img = extracting_colour_names_from_hex(hex_colours_img)

		colour0_match_df = dataset_df[dataset_df.eq(CSS_colours_img[0]).any(1)]
		colour1_match_df = dataset_df[dataset_df.eq(CSS_colours_img[1]).any(1)]
		colour2_match_df = dataset_df[dataset_df.eq(CSS_colours_img[2]).any(1)]
		colour3_match_df = dataset_df[dataset_df.eq(CSS_colours_img[3]).any(1)]
		colour4_match_df = dataset_df[dataset_df.eq(CSS_colours_img[4]).any(1)]

		colour_matches_df = pd.concat([colour0_match_df,
		                               colour1_match_df,
		                               colour2_match_df,
		                               colour3_match_df,
		                               colour4_match_df])

		colour_matches_df.drop_duplicates(inplace=True)

		get_artworks_with_similar_colours(colour_matches_df, placeholder)

		if st.button('Generate other recommendations'):
			refresh_button(placeholder)


if __name__ == '__main__':
	main()
