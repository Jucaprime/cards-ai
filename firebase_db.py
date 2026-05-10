import firebase_admin
from firebase_admin import credentials, firestore

# conectar
cred = credentials.Certificate("firebase_key.json")

firebase_admin.initialize_app(cred)

db = firestore.client()

# salvar teste
db.collection("testes").add({

    "nome": "teste firebase",
    "status": "funcionando"

})

print("Dado salvo!")