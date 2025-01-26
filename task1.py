import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ========================================================================================================
# I am going to list my steps here. Normally it is used with the code but nowadays AI follow the same style for 
# code comments, So I will list here
# ========================================================================================================

# Step 1: Load the dataset
# Step 2: Define features (X) and target variable (y)
# Step 3: Data Preprocessing (Check for missing values)
# Step 4: Split the data into training and testing sets (I am using 80% of the data for training and 20% for testing)
# Step 5: Initialize the Random Forest Classifier (I choosen Random Forest instead of Logistic Regression because I
#         think it will give better results)
# Step 6: Train the model
# Step 7: Make predictions on the test set
# Step 8: Evaluate the model (Calculate [Accuracy, Precision, Recall, etc]) I have generated classification report
#         and confusion matrix.
# Step 9: Save the model
# ========================================================================================================


df = pd.read_csv('cricket_dataset.csv')

if df.isnull().sum().any():
    print("Warning: Missing values found in the dataset. Removing rows having missing values")
    df.dropna(inplace=True)
else:
    print("No missing values found. Proceeding to model training.")


X = df[['total_runs', 'wickets', 'target', 'balls_left']]  
y = df['won']  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)


print("Training the Random Forest model...")
model.fit(X_train, y_train)
print("Model training complete!")

print("Making predictions on the test set...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Generate and save the confusion matrix as an image
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('imgs/confusion_matrix.png')  
plt.close()

print("Confusion matrix saved as 'confusion_matrix.png' in imgs directory")

joblib.dump(model, 'model.pkl')
print("\nModel saved as 'model.pkl'")