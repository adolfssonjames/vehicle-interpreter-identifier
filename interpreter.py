from sklearn.tree import DecisionTreeClassifier  #Importerar DesicionTreeClassifier från sklearn
from sklearn.preprocessing import LabelEncoder  #Importerar LabelEncoder från sklearn
import numpy as np  #Importerar numpypaketet
import json

file_name = "vehicle_data.json" #Filnamn för json-filen som innehåller vårt dataset
with open(file_name, "r+") as json_file: #Öppnar json-filen för att läsa in datan kan även write pga r+
    data = json.load(json_file) #Laddar in datan från json-filen
    print("Data loaded from", file_name)
    
#Konverterar dictionaryn till två separata listor för att kunna använda datan i vår klassificerare
X = np.array([[key] for key in data.keys()]) #Skapar en array med antal hjul som nycklar
y = np.array(list(data.values())) #Skapar en array med fordonstyperna som värden

#Encodar variabeln y så att vi kan använda den i vår klassificerare
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y) #fit_transform används för att konvertera datan

# Skapar en klassificerare och tränar den med vårt dataset
clf = DecisionTreeClassifier()
clf.fit(X, y_encoded)

#Promtar användaren att ange antal hjul på sitt fordon
while True:
    num_wheels = int(input("Enter the number of wheels of your vehicle: "))
    if num_wheels == 1 or num_wheels == 5 or num_wheels > 6:
        print("Invalid input. Please enter a valid number of wheels.")
    else:
        break

#Förbereder inputdatan för att göra en prediction
input_data = np.array([[num_wheels]]) #skapar en array med antal hjul som input

#Predictions
prediction_encoded = clf.predict(input_data) #Gör en prediction med vår klassificerare
predicted_vehicle_type = label_encoder.inverse_transform(prediction_encoded) #Konverterar den encodade predictionen tillbaka till den ursprungliga formen
print("Predicted vehicle type:", predicted_vehicle_type[0])

#Promtar användaren att beskriva problemet med sitt fordon
vehicle_issue = input("Describe the issue with your vehicle, if it has any: ")

#Anger om fordonet är i kritiskt skick eller inte baserat på användarens input
if any(keyword in vehicle_issue.lower() for keyword in ["critical", "sounds", "sound", "yellow light", "red light", "flat", "slow", "powerless", "no power", "engine sound", "engine noise", "noise", "smell", "bad engine", "weak", "leak", "leaking", "hot", "not running", "overheat", "overheats", "overheating"]):
    print("Your vehicle is in critical or bad condition. Visit a mechanic immediately.")
else:
    print("Your vehicle is all good. No need for service. Happy driving!")

