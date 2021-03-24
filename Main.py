import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from flask import Flask,render_template,request




# import given data
DataCsv = pd.read_csv('data.csv')

# Convert CSV data into Excel format
DataCsv.to_excel('CancerData.xlsx', index=None, header=True)
DataExcel = pd.read_excel('CancerData.xlsx')

# Working on CSV files
# Drop last column having values of no use
DataCsv = DataCsv.dropna(axis=1)

# Count Malignant and Benign patients
# print(DataCsv['diagnosis'].value_counts())

# Encode the categorical values
# Malignant->1
# Benign->0
for i in range(0, len(DataCsv)):
    if DataCsv.iloc[i, 1] == 'M':
        DataCsv.iloc[i, 1] = float(1)
    elif DataCsv.iloc[i, 1] == 'B':
        DataCsv.iloc[i, 1] = float(0)

# Split our data into independent (x->2-31) and Dependent (y->1) Data sets
x = DataCsv.iloc[:, 2:32].values
y = DataCsv.iloc[:, 1].values

# Let's Train our model on 95% of data
# We can also Test the model with remaining 5% of data
TrainX, TestX, TrainY, TestY = train_test_split(x, y, test_size=0.05)
# Validating Data
TrainX = np.array(TrainX)
TrainY = np.array(TrainY)
TrainX = TrainX.astype(np.float64)
TrainY = TrainY.astype(np.float64)

TestX = np.array(TestX)
TestY = np.array(TestY)
TestX = TestX.astype(np.float64)
TestY = TestY.astype(np.float64)

# Data Normalization
'''for row in range(len(TrainX)):
    Xmax=np.max(TrainX[row,:])
    Xmin=np.min(TrainX[row,:])
    for col in range(29):
        X=TrainX[row][col]
        X_train[row][col]=(X-Xmin)/(Xmax-Xmin)'''

# Use SVM modelling to train the data
svc_lin=SVC(kernel='linear',random_state=0)
svc_lin.fit(TrainX,TrainY)
Training_Accuracy=svc_lin.score(TrainX,TrainY)


Testing_Accuracy=svc_lin.score(TestX,TestY)

# Web App
Main=Flask(__name__)
@Main.route('/')
def design():
    return render_template('design.html')

@Main.route('/', methods=['POST'])
def getvalue():
    Radiusmean=request.form['radiusmean']
    Texturemean = request.form['texturemean']
    Perimetermean = request.form['perimetermean']
    Areamean=request.form['areamean']
    Smoothnessmean=request.form['smothnessmean']
    Compactnessmean = request.form['compactnessmean']
    Concativiytymean = request.form['concativitymean']
    Concavepointmean = request.form['concavepointmean']
    Symmetrymean = request.form['symmetrymean']
    FractalDimensionMean = request.form['fractaldimensionmean']

    RadiusSe = request.form['radiusse']
    Texturese=request.form['Texturese']
    Perimeterse = request.form['Perimeterse']
    AreaSE = request.form['arease']
    SmoothnessSE=request.form['smoothnessse']
    compactnessSE=request.form['compactnessse']
    ConcativiytySE = request.form['concativityse']
    concavepointse=request.form['concavepointse']
    Symmetryse=request.form['symmetryse']
    FractalDimensionSE = request.form['fractaldimensionse']

    Radiusworst=request.form['radiusworst']
    TextureWorst=request.form['textureworst']
    PerimeterWorst=request.form['perimeterworst']
    AreaWorst=request.form['areaworst']
    SmoothnessWorst = request.form['smoothnessworst']
    CompactnessWorst=request.form['compactnessworst']
    ConcativtyWorst = request.form['concativtyworst']
    Concavepointworst=request.form['concavepointworst']
    SymmetryWorst = request.form['symmetryworst']
    FractalDimansionWorst = request.form['fractaldimansionworst']

    X_test=[[Radiusmean,Texturemean,Perimetermean,Areamean,Smoothnessmean,Compactnessmean,
             Concativiytymean,Concavepointmean,Symmetrymean,FractalDimensionMean,
             RadiusSe,Texturese,Perimeterse,AreaSE,SmoothnessSE,compactnessSE,ConcativiytySE,
             concavepointse,Symmetryse,FractalDimensionSE,
             Radiusworst,TextureWorst,PerimeterWorst,AreaWorst,SmoothnessWorst,
             CompactnessWorst,ConcativtyWorst,Concavepointworst,SymmetryWorst,FractalDimansionWorst]]

    result = svc_lin.predict(X_test)
    print(result[0])
    if result==1:
        return render_template("ResultMal.html", RESULT=result)
    if result==0:
        return render_template("ResultBen.html", RESULT=result)




if __name__ == '__main__':
    Main.run(debug=True)
