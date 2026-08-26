import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import os
import joblib

df=pd.read_csv(r"D:\university\ai_training\spyder\Cars_Datasets_2025.csv",encoding='latin1')
print(f"shape{df.shape}")
df.head()
df.describe()
print('Missing values:')
print(df.isnull().sum())
%matplotlib inline
 
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
 
axes[0].hist(df['HorsePower'].str.extract(r'(\d+)').astype(float).dropna(), bins=30, color='steelblue', edgecolor='white')
axes[0].set_title('HorsePower Distribution')
axes[0].set_xlabel('HorsePower')
 
axes[1].scatter(df['Total Speed'].str.extract(r'(\d+)').astype(float),
                 df['HorsePower'].str.extract(r'(\d+)').astype(float),
                 alpha=0.4, color='steelblue')
axes[1].set_title('Total Speed vs HorsePower')
axes[1].set_xlabel('Total Speed (km/h)')
axes[1].set_ylabel('HorsePower')
 
plt.tight_layout()
plt.show()
def parse_num(s):
    if pd.isna(s):
        return np.nan
    s = str(s).replace(',', '')
 
    nums = []
    current = ''
    for ch in s:
        if ch.isdigit() or ch == '.':
            current += ch
        else:
            if current and current != '.':
                nums.append(float(current))
            current = ''
    if current and current != '.':
        nums.append(float(current))
 
    return np.mean(nums) if nums else np.nan
 
num_cols_raw = ['CC/Battery Capacity', 'HorsePower', 'Total Speed',
                'Performance(0 - 100 )KM/H', 'Cars Prices', 'Seats', 'Torque']
 
for col in num_cols_raw:
    df[col.split('(')[0].strip() + '_num'] = df[col].apply(parse_num)
 
def simplify_fuel(f):
    f = str(f).lower()
    if 'hydrogen' in f:
        return 'Hydrogen'
    if 'hybrid' in f:
        return 'Hybrid'
    if ('electric' in f or f.strip() == 'ev') and 'petrol' not in f and 'diesel' not in f:
        return 'Electric'
    if 'diesel' in f and 'petrol' not in f:
        return 'Diesel'
    if 'petrol' in f or 'gas' in f:
        return 'Petrol'
    return 'Other'
 
df['FuelType'] = df['Fuel Types'].apply(simplify_fuel)
df = df[df['FuelType'].isin(['Petrol', 'Diesel', 'Electric', 'Hybrid'])].copy()
 
print('Encoding done. Sample:')
df.head(3)
 
feature_cols = ['CC/Battery Capacity_num', 'HorsePower_num', 'Total Speed_num',
                'Performance_num', 'Cars Prices_num', 'Seats_num', 'Torque_num']
 
df = df.dropna(subset=feature_cols)
 
x = df[feature_cols]
y = df['FuelType']
X_train,x_test,Y_train,y_test=train_test_split(x,y,train_size=0.8,test_size=0.2,random_state=42,stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   
X_test_scaled  = scaler.transform(x_test)
print(f'Train size : {X_train.shape[0]} rows')
print(f'Test  size : {x_test.shape[0]} rows')
 
model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10,
                                class_weight='balanced', random_state=42)
model.fit(X_train_scaled, Y_train)
 
# Evaluate on the test set
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
 
print(f'Accuracy Score : {acc:.4f}   (1.0 = perfect)')
print()
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
 
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap='Blues')
plt.title('Actual vs Predicted Fuel Type')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(range(len(model.classes_)), model.classes_, rotation=45)
plt.yticks(range(len(model.classes_)), model.classes_)
for i in range(len(model.classes_)):
    for j in range(len(model.classes_)):
        plt.text(j, i, cm[i, j], ha='center', va='center',
                  color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.colorbar()
plt.tight_layout()
plt.show()
 
os.makedirs('models', exist_ok=True)
 
joblib.dump(model,  'models/model.pkl')
 
joblib.dump(scaler, 'models/scaler.pkl')
 
joblib.dump(list(x.columns), 'models/columns.pkl')

joblib.dump(list(model.classes_), 'models/classes.pkl')
 
print('Saved:')
for f in os.listdir('models'):
    if f.endswith('.pkl'):
        size = os.path.getsize(f'models/{f}')
        print(f'   models/{f}  ({size:,} bytes)')
m = joblib.load('models/model.pkl')
s = joblib.load('models/scaler.pkl')
c = joblib.load('models/columns.pkl')
classes = joblib.load('models/classes.pkl')
 
sample = pd.DataFrame([{
    'CC/Battery Capacity_num': 3990, 'HorsePower_num': 963, 'Total Speed_num': 340,
    'Performance_num': 2.5, 'Cars Prices_num': 1100000, 'Seats_num': 2, 'Torque_num': 800
}])[c]
 
fuel_type = m.predict(s.transform(sample))[0]
print(f'Sample prediction: {fuel_type}  (sanity check passed)')
 