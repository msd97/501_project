import firebase_admin
from firebase_admin import db

default_app = firebase_admin.initialize_app(options={'databaseURL':"https://artcompanion-8adad-default-rtdb.firebaseio.com/"})
ref = db.reference("/others/")

ref.update({"detection":'"Albrecht Dürer"'})
print(ref.child("detection").get())