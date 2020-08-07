from flask import Flask,render_template,request
import pickle

app=Flask(__name__)

file=open("model.pkl","rb")
LR=pickle.load(file)
file.close()

@app.route("/" , methods=["GET","POST"])
def home():
    if request.method == "POST":
        myDict=request.form
        age=float(myDict["age"])
        bmi=float(myDict["bmi"])
        insulin = float(myDict["insulin"])
        glucose = float(myDict["glucose"])
        BP = float(myDict["BP"])
        thickness = float(myDict["thickness"])
        pred = [age, bmi, insulin, glucose, BP, thickness]
        Diab_pred = LR.predict_proba([pred])[0][1] * 100
        print(Diab_pred)
        return render_template("result.html",age=age,bmi=bmi,insulin=insulin,glucose=glucose,BP=BP,thickness=thickness,res=round(Diab_pred))

    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)