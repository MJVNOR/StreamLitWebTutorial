#Empezaremos bajando las librerías necesarias
#Libreria para usar los componentes de Streamlit
import streamlit as st

#Libreria para poder desempaquetar nuestro modelo
import pickle
from pickle import dump

#Con st.title creamos un título para nuestra aplicación 
st.title('Como subir GRATIS a la Web tus modelos de inteligencia artificial')

#Dentro de la librería tenemos la opción de poder escribir con markdown
st.markdown('''
Este es un ejemplo para poder usar Streamlit con un modelo que clasifica texto según que sí está en contra o a favor del gobierno Mexicano.
''')

#Aquí desempacamos nuestro modelo
nombreArchivo = 'ClasificadorTweets.pkl'
modeloCargado = pickle.load(open(nombreArchivo, 'rb'))

#Lo guardo en una dupla porque yo guarde mi modelo y para poder convertir el texto utilizo Tfidf
ClasificadorTweets, Tfidf_vect = modeloCargado

#Con st.text_input hacemos un widget para tener un input del usuario
input = st.text_input(label='Inserte su texto', value='Amlo presidente')

#Convertimos el input a Tdfidf
prediccionEjemplo = Tfidf_vect.transform([input])

resultadoPorcentaje = ClasificadorTweets.predict_proba(prediccionEjemplo)

aux = []
for i in resultadoPorcentaje:
    for j in i:
        conservador = j
        aux = [j]
if conservador <= 0.5:
    #Con st.write mostramos texto en pantalla
    st.write('Resultado: A favor del gobierno')
else:
    st.write('Resultado: En contra del gobierno')
st.write('Result: {}'.format(resultadoPorcentaje))