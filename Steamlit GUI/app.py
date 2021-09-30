import streamlit as st 
import joblib,os
import numpy as np
#import spacy_streamlit
import spacy
import pandas as pd
nlp = spacy.load("en_core_web_sm")
import matplotlib.pyplot as plt 
import matplotlib
import itertools
from PIL import Image
#from spacy import displacy
matplotlib.use("Agg")
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



# Vectorizer using TFIDF
news_vectorizer = open(r"C:\Users\Pro\Downloads\StreamlitGUIproject\vectorTF.pkl","rb")
news_cv = joblib.load(news_vectorizer)

def load_prediction_models(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model


# Get the Keys
def get_keys(val,my_dict):
	for key,value in my_dict.items():
		if val == value:
			return key


def main():
	"""News Classifier"""
	st.title("SOCIAL MEDIA NEWS CLASSFIER ")
	# st.subheader("Machine Learning Classifier")
	html_temp = """
	<div style="background-color:#2a2a72;padding:10px">
	<h1 style="color:white;text-align:center;">Machine Learning Classifier </h1>
	</div>
	"""
	st.markdown(html_temp,unsafe_allow_html=True)

	activity = ['Prediction','NLP']
	choice = st.sidebar.selectbox("Operation Used",activity)


	if choice == 'Prediction':
		st.info("Prediction with ML")

		news_text = st.text_area("Enter News Here","Type Here")
		all_ml_models = ["LR","RFOREST","MB","DECISION_TREE","GB","SVM"]
		model_choice = st.selectbox("Select Model",all_ml_models)

		prediction_labels = {'fake': 0,'reel': 1}
		if st.button("Classify"):
			st.text("Original Text::\n{}".format(news_text))
			vect_text = news_cv.transform([news_text]).toarray()
			if model_choice == 'LR':
				predictor = load_prediction_models(r"C:\Users\Pro\Downloads\StreamlitGUIproject\Pickle_LR_Model.pkl")
				prediction = predictor.predict(vect_text)
				# st.write(prediction)
			elif model_choice == 'RFOREST':
				predictor = load_prediction_models(r"C:\Users\Pro\Downloads\StreamlitGUIproject\Pickle_RFM_Model.pkl")
				prediction = predictor.predict(vect_text)
				# st.write(prediction)
			elif model_choice == 'MB':
				predictor = load_prediction_models(r"C:\Users\Pro\Downloads\StreamlitGUIproject\Pickle_MB_Model.pkl")
				prediction = predictor.predict(vect_text)
				# st.write(prediction)
			elif model_choice == 'GB':
				predictor = load_prediction_models(r"C:\Users\Pro\Downloads\StreamlitGUIproject\Pickle_GB_Model.pkl")
				prediction = predictor.predict(vect_text)
				# st.write(prediction)
			elif model_choice == 'SVM':
				predictor = load_prediction_models(r"C:\Users\Pro\Downloads\StreamlitGUIproject\Pickle_SVM_Model.pkl")
				prediction = predictor.predict(vect_text)
				# st.write(prediction)
			elif model_choice == 'DECISION_TREE':
				predictor = load_prediction_models(r"C:\Users\Pro\Downloads\StreamlitGUIproject\Pickle_DTC_Model.pkl")
				prediction = predictor.predict(vect_text)
				# st.write(prediction)

			final_result = get_keys(prediction,prediction_labels)
			st.success("News Categorized as:: {}".format(final_result))

	if choice == 'NLP':
		st.info("Natural Language Processing of Text")
		raw_text = st.text_area("Enter News Here","Type Here")
		nlp_task = ["Tokenization","Lemmatization","NER","POS Tags"]
		task_choice = st.selectbox("Choose NLP Task",nlp_task)
		if st.button("Analyze"):
			st.info("Original Text::\n{}".format(raw_text))

			docx = nlp(raw_text)
			if task_choice == 'Tokenization':
				result = [token.text for token in docx ]
			elif task_choice == 'Lemmatization':
				result = ["'Token':{},'Lemma':{}".format(token.text,token.lemma_) for token in docx]
			elif task_choice == 'NER':
				result = [(entity.text,entity.label_)for entity in docx.ents]
			elif task_choice == 'POS Tags':
				result = ["'Token':{},'POS':{},'Dependency':{}".format(word.text,word.tag_,word.dep_) for word in docx]

			st.json(result)

		if st.button("Tabulize"):
			docx = nlp(raw_text)
			c_tokens = [token.text for token in docx ]
			c_lemma = [token.lemma_ for token in docx ]
			c_pos = [token.pos_ for token in docx ]

			new_df = pd.DataFrame(zip(c_tokens,c_lemma,c_pos),columns=['Tokens','Lemma','POS'])
			st.dataframe(new_df)



		if st.checkbox("WordCloud"):
				c_text = raw_text
				#mask = np.array(Image.open(r"cloud.png"))
				wordcloud = WordCloud().generate(c_text)
				plt.imshow(wordcloud,interpolation='bilinear')
				plt.axis("off")
				st.pyplot()




	about = st.sidebar.subheader("About")
	credit = st.sidebar.info("Giving more insights about the nature of the words used in the tweets + Predict the outcome as Reel or Fake.  ")
 	




if __name__ == '__main__':
	main()
