from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def build_and_train_models(X_train, y_train, X_test, y_test):
    # Define models
    models = {
        "Neural Network": Sequential([
            Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')
        ]),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, random_state=42)
    }

    # Compile Neural Network
    models["Neural Network"].compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train models
    print("Training models...")
    models["Neural Network"].fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=0)
    models["Random Forest"].fit(X_train, y_train)
    models["SVM"].fit(X_train, y_train)

    # Evaluate models and compare accuracy
    best_model = None
    best_accuracy = 0
    model_accuracies = {}

    for name, model in models.items():
        if name == "Neural Network":
            _, accuracy = models[name].evaluate(X_test, y_test, verbose=0)
        else:
            accuracy = accuracy_score(y_test, model.predict(X_test))

        model_accuracies[name] = accuracy
        print(f"{name} Accuracy: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    print("\nBest Model:")
    print(f"{best_model} with Accuracy: {best_accuracy:.4f}")

    # Return the best model and accuracies for further use
    return best_model, model_accuracies
