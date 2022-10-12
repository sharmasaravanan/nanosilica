# importing lib
import os
import random
import warnings
import matplotlib
matplotlib.use("TKAgg")

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from model import MyModel, evaluation

warnings.filterwarnings('ignore')

# loading dataset
dataset = pd.read_excel("dataset.xlsx")

# displaying top-5 rows and properties of the dataset
print("Top-5 rows of the dataset")
display(dataset.head())

row, col = dataset.shape
print(f"\nThe number of rows and columns in the dataset: {row} and {col}")

print('\nDescriptive analysis of the dataset')
descDataset = dataset.describe()
display(descDataset)

print(f'\nColumns in the datasets are : {list(dataset.columns)}')

# synthetic data generation to avoid under fitting
processedDataset = pd.DataFrame(columns=dataset.columns)
k = 0
for col in dataset.columns:
    mean = descDataset[col]['mean']
    std = descDataset[col]['std']
    processedDataset[col] = list(map(lambda x: (random.random() * std + mean + k), range(10000)))

processedDataset = pd.concat([processedDataset, dataset], ignore_index=True)

row, col = processedDataset.shape
print(f"\nThe number of rows and columns in the generated dataset: {row} and {col}")

print('\nDescriptive analysis of the generated dataset')
display(processedDataset.describe())

# checking for the missing values
print("\nChecking for the missing values")
print(processedDataset.isnull().sum(axis=0).reset_index())

# checking for the missing values post preprocessing
print('\nDisplaying columns with number of count post preprocessing')
display(dataset.isna().any())

# data visualization
sns.pairplot(processedDataset)
plt.show()

# correlation of the dataset
corr = processedDataset.corr()
plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, cmap='Blues')
b, t = plt.ylim()
plt.ylim(b + 0.5, t - 0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# distribution of the dataset
ax = sns.distplot(processedDataset.CompressiveStrength)
ax.set_title("Compressive Strength Distribution")
plt.show()

_, ax = plt.subplots(figsize=(10, 7))
sns.scatterplot(y="CompressiveStrength", x="Cement", hue="nanoSilica", size="CoarseAggregates", data=processedDataset,
                ax=ax, sizes=(50, 300))
ax.set_title("CC Strength vs (Cement, nanoSilica, CoarseAggregates)")
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.show()

_, ax = plt.subplots(figsize=(10, 7))
sns.scatterplot(y="CompressiveStrength", x="CoarseAggregates", hue="FineAggregates", size="nanoSilica",
                data=processedDataset, ax=ax, sizes=(50, 300))
ax.set_title("CC Strength vs (Coarse aggregate, Fine Aggregates, nanoSilica)")
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.show()

# saving the preprocessed dataset
processedDataset.to_csv('preprocessedDataset.csv', index=False)

# separating features
x = processedDataset.iloc[:, :6]
y = processedDataset.iloc[:, 6]

print(f"\nThe shape of the input and output data are {x.shape},{y.shape}")

# scaling and normalization the dataset
scaler = MinMaxScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)

if not os.path.isdir('model'):
    os.mkdir(os.path.join('model'))
joblib.dump(scaler, open(os.path.join('model', 'scaler.pkl'), 'wb'))

# Splitting dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.15, random_state=3)

print(f"The shapes of the train dataset is {X_train.shape}")
print(f"The shapes of the test dataset is {X_test.shape}")

# defining the model
n_inputs, n_outputs = X_train.shape[1], 1
model = MyModel(n_inputs, n_outputs)

# Training the model
history = model.fit(X_train, y_train, epochs=100, batch_size=256, validation_split=0.1)

# plotting the training  metrics
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# saving model
if not os.path.isdir('model'):
    os.mkdir(os.path.join('model'))

model.save('nanoSilica.h5')

# evaluating model
evaluation(model, X_test, y_test)
