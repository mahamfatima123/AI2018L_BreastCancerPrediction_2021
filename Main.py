import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from flask import Flask,render_template,request




# import CSV data
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

# Train the model on 75% of our data
# Test the model with remaining 25% of the data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.05)
print(X_train)
# Validating Data
X_train = np.array(X_train)
y_train = np.array(y_train)
X_train = X_train.astype(np.float64)
y_train = y_train.astype(np.float64)

X_test = np.array(X_test)
y_test = np.array(y_test)
X_test = X_test.astype(np.float64)
y_test = y_test.astype(np.float64)

# Data Normalization
'''for row in range(len(X_train)):
    Xmax=np.max(X_train[row,:])
    Xmin=np.min(X_train[row,:])
    for col in range(29):
        X=X_train[row][col]
        X_train[row][col]=(X-Xmin)/(Xmax-Xmin)'''

# Use SVM modelling to train the data
svc_lin=SVC(kernel='linear',random_state=0)
svc_lin.fit(X_train,y_train)
print('Training Accuracy:',svc_lin.score(X_train,y_train))


# print('Testing Accuracy:',svc_lin.score(X_test,y_test))

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
             CompactnessWorst,ConcativtyWorst,Concavepointworst,SymmetryWorst,FractalDimansionWorst],

            [Radiusmean, Texturemean, Perimetermean, Areamean, Smoothnessmean, Compactnessmean,
             Concativiytymean, Concavepointmean, Symmetrymean, FractalDimensionMean,
             RadiusSe, Texturese, Perimeterse, AreaSE, SmoothnessSE, compactnessSE, ConcativiytySE,
             concavepointse, Symmetryse, FractalDimensionSE,
             Radiusworst, TextureWorst, PerimeterWorst, AreaWorst, SmoothnessWorst,
             CompactnessWorst, ConcativtyWorst, Concavepointworst, SymmetryWorst, FractalDimansionWorst]]
    y_test=[1,0]
    result=(svc_lin.score(X_test, y_test))
    print('Testing Accuracy:', svc_lin.score(X_test, y_test))
    return render_template('pass.html',
                           RM=Radiusmean,
                           TM=Texturemean,
                           PM=Perimetermean,
                           AM=Areamean ,
                           SM=Smoothnessmean,
                           CM=Compactnessmean,
                           ConMN=Concativiytymean,
                           conpointmean=Concavepointmean,
                           symMn=Symmetrymean,
                           FM=FractalDimensionMean,

                           radiusSE=RadiusSe,
                           Texse=Texturese,
                           PerSE=Perimeterse,
                           ArSe=AreaSE ,
                           smoSE=SmoothnessSE,
                           compSE=compactnessSE,
                           ConSe=ConcativiytySE,
                           concvptse=concavepointse,
                           SymSE=Symmetryse,
                           FracDimSE=FractalDimensionSE,

                           RW=Radiusworst,
                           TW=TextureWorst,
                           PW=PerimeterWorst,
                           AW=AreaWorst,
                           SmoWorst=SmoothnessWorst,
                           CompW=CompactnessWorst,
                           ConWrst=ConcativtyWorst,
                           ConptW=Concavepointworst,
                           SymmWrst=SymmetryWorst,
                           FracDimWrst=FractalDimansionWorst)

@Main.route('/Result')
def Result():
    Radiusmean = request.form['radiusmean']
    Texturemean = request.form['texturemean']
    Perimetermean = request.form['perimetermean']
    Areamean = request.form['areamean']
    Smoothnessmean = request.form['smothnessmean']
    Compactnessmean = request.form['compactnessmean']
    Concativiytymean = request.form['concativitymean']
    Concavepointmean = request.form['concavepointmean']
    Symmetrymean = request.form['symmetrymean']
    FractalDimensionMean = request.form['fractaldimensionmean']

    RadiusSe = request.form['radiusse']
    Texturese = request.form['Texturese']
    Perimeterse = request.form['Perimeterse']
    AreaSE = request.form['arease']
    SmoothnessSE = request.form['smoothnessse']
    compactnessSE = request.form['compactnessse']
    ConcativiytySE = request.form['concativityse']
    concavepointse = request.form['concavepointse']
    Symmetryse = request.form['symmetryse']
    FractalDimensionSE = request.form['fractaldimensionse']

    Radiusworst = request.form['radiusworst']
    TextureWorst = request.form['textureworst']
    PerimeterWorst = request.form['perimeterworst']
    AreaWorst = request.form['areaworst']
    SmoothnessWorst = request.form['smoothnessworst']
    CompactnessWorst = request.form['compactnessworst']
    ConcativtyWorst = request.form['concativtyworst']
    Concavepointworst = request.form['concavepointworst']
    SymmetryWorst = request.form['symmetryworst']
    FractalDimansionWorst = request.form['fractaldimansionworst']

    X_test = [[Radiusmean, Texturemean, Perimetermean, Areamean, Smoothnessmean, Compactnessmean,
               Concativiytymean, Concavepointmean, Symmetrymean, FractalDimensionMean,
               RadiusSe, Texturese, Perimeterse, AreaSE, SmoothnessSE, compactnessSE, ConcativiytySE,
               concavepointse, Symmetryse, FractalDimensionSE,
               Radiusworst, TextureWorst, PerimeterWorst, AreaWorst, SmoothnessWorst,
               CompactnessWorst, ConcativtyWorst, Concavepointworst, SymmetryWorst, FractalDimansionWorst],

              [Radiusmean, Texturemean, Perimetermean, Areamean, Smoothnessmean, Compactnessmean,
               Concativiytymean, Concavepointmean, Symmetrymean, FractalDimensionMean,
               RadiusSe, Texturese, Perimeterse, AreaSE, SmoothnessSE, compactnessSE, ConcativiytySE,
               concavepointse, Symmetryse, FractalDimensionSE,
               Radiusworst, TextureWorst, PerimeterWorst, AreaWorst, SmoothnessWorst,
               CompactnessWorst, ConcativtyWorst, Concavepointworst, SymmetryWorst, FractalDimansionWorst]]
    y_test = [1, 0]
    result = (svc_lin.score(X_test, y_test))
    print('Testing Accuracy:', svc_lin.score(X_test, y_test))
    return render_template("Result.html",RESULT=result)


if __name__ == '__main__':
    Main.run(debug=True)
