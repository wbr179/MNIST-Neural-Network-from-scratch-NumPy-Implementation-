import os
import gzip
import idx2numpy
import numpy as np
from PIL import Image
from random import randint
import matplotlib.pyplot as plt

# Lernrate global definiert:
learning_rate = 0.1  # als Startwert

def load_mnist_dataset(folder_path):
    # Dateinamen (lokal gespeichert)
    fname_train_images = os.path.join(folder_path, "train-images-idx3-ubyte.gz")
    fname_train_labels = os.path.join(folder_path, "train-labels-idx1-ubyte.gz")
    fname_test_images = os.path.join(folder_path, "t10k-images-idx3-ubyte.gz")
    fname_test_labels = os.path.join(folder_path, "t10k-labels-idx1-ubyte.gz")

    # Lade die Bilder und Labels mit gzip + idx2numpy
    with gzip.open(fname_train_images, 'rb') as f:
        train_images = idx2numpy.convert_from_file(f)
    with gzip.open(fname_train_labels, 'rb') as f:
        train_labels = idx2numpy.convert_from_file(f)
    with gzip.open(fname_test_images, 'rb') as f:
        test_images = idx2numpy.convert_from_file(f)
    with gzip.open(fname_test_labels, 'rb') as f:
        test_labels = idx2numpy.convert_from_file(f)

    return (train_images, train_labels), (test_images, test_labels)

def prepare_data(X_raw, y_raw, flatten=True, normalize=True):
    """
    Eingabeschicht - Eingang der Daten

    Bereitet Bild- und Labeldaten für das Training vor.
    
    Parameter:
    - X_raw: Array der Bilddaten 
    - y_raw: Array der Labels 
    - flatten: ob die Bilder in 1D-Vektoren (784,) umgewandelt werden sollen
    - normalize: ob Pixelwerte in den Bereich [0.0, 1.0] gebracht werden sollen

    Rückgabe:
    - X: verarbeitete Bilddaten
    - y: Labels als int32
    """
    X = X_raw.astype(np.float32)
    y = y_raw.astype(np.int32)

    if normalize:
        X = X / 255.0

    if flatten:
        # Bilder in Vektor umwandeln
        # im Beispiel mit MNIST 28x28 Pixel: Vektor mit Länge 784
        X = X.reshape(X.shape[0], -1)

    return X, y

def sigmoid(z):
    # Sigmoid-Aktivierungsfunktion

    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    # Ableitung der Sigmoid-Funktion (für Backpropagation)

    s = sigmoid(z)
    return s * (1 - s)

def init_params(input_size=784, hidden_size=64, output_size=10):
    # Initialisiert Gewichte und Biases zufällig

    np.random.seed(42)  # Reproduzierbarkeit
    
    # Gewichte klein initialisieren
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    
    return W1, b1, W2, b2

def forward_pass(X, W1, b1, W2, b2):
    # Berechnet Vorwärtsdurchlauf durch das Netz
    
    # versteckte Verarbeitungs-Schicht:
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    
    # Ausgabeschicht: Vorhersage treffen
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    
    return Z1, A1, Z2, A2

def mse_calc(y_true, y_pred):
    # Berechnet den mittleren quadratischen Fehler (MSE) als Loss-Funktion
    # y_true: One-Hot-Vektor der Soll-Ausgabe (z.B. [0,0,1,0,...])
    # y_pred: Vorhersage des Netzes (z.B. [0.1,0.05,0.9,...])
    
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true, y_pred):
    # Ableitung der MSE-Funktion nach den Vorhersagen.
    # Wird für Backpropagation benötigt.
    
    return (y_pred - y_true)

def backprop(X, y, Z1, A1, Z2, A2, W1, b1, W2, b2, learning_rate):
    # führt Backpropagation durch und aktualisiert die Parameter

    # Output layer Fehler (Abweichung zwischen Vorhersage und Soll)
    dZ2 = (A2 - y) * sigmoid_derivative(Z2)   # (batch, 10)
    # Gradient für W2 und b2
    dW2 = np.dot(A1.T, dZ2) / X.shape[0]      # Mittelwert über Batch
    db2 = np.sum(dZ2, axis=0, keepdims=True) / X.shape[0]

    # Hidden layer Fehler
    dZ1 = np.dot(dZ2, W2.T) * sigmoid_derivative(Z1)  # (batch, 64)
    # Gradient für W1 und b1
    dW1 = np.dot(X.T, dZ1) / X.shape[0]
    db1 = np.sum(dZ1, axis=0, keepdims=True) / X.shape[0]

    # Gewichte und Biases anpassen
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    return W1, b1, W2, b2

def evaluate_accuracy(X, y, W1, b1, W2, b2):
    # berechnet Klassifizierungsgenauigkeit (Accuracy)
    # return entspricht dem Anteil der richtig erkannten Bilder
    correct = 0
    total = X.shape[0]

    for i in range(total):
        # Eingabe in richtige Form bringen
        x_input = X[i].reshape(1, -1)

        # forward pass aufrufen
        _, _, _, A2 = forward_pass(x_input, W1, b1, W2, b2)

        # Vorhersage:
        predicted = np.argmax(A2)  # Index der höchsten Aktivierung

        # Vergleich mit Label
        if predicted == y[i]:
            correct += 1

    return correct / total

def plot_training(history_loss, history_acc):
    # Visualisiert den Trainingsverlauf.

    plt.figure(figsize=(10,4))

    # Loss-Verlauf
    plt.subplot(1,2,1)
    plt.plot(history_loss, label="Loss", marker="o")
    plt.xlabel("Epoche")
    plt.ylabel("MSE Loss")
    plt.title("Fehler (Loss) pro Epoche")
    plt.legend()

    # Accuracy-Verlauf
    plt.subplot(1,2,2)
    plt.plot(history_acc, label="Test Accuracy", marker="o", color="green")
    plt.xlabel("Epoche")
    plt.ylabel("Accuracy")
    plt.title("Genauigkeit auf Testdaten pro Epoche")
    plt.legend()

    plt.tight_layout()
    plt.show()

def show_misclassified(X, y, W1, b1, W2, b2, max_samples=10):
    # fehlerhaft klassifizierte Bilder anzeigen

    shown = 0

    for i in range(len(X)):

        # Abbruch wenn mehr als max_samples Bilder gezeigt wurden
        if shown >= max_samples:
            break

        x_input = X[i].reshape(1, -1)
        _, _, _, A2 = forward_pass(x_input, W1, b1, W2, b2)
        predicted = np.argmax(A2)
        true_label = y[i]

        if predicted != true_label:
            # bei fehlerhafter Vorhersage, Bild zeigen:

            plt.imshow(X[i].reshape(28,28), cmap="gray")
            plt.title(f"Richtig: {true_label}, Vorhersage: {predicted}")
            plt.axis("off")
            plt.show()
            shown += 1

def train_model(X_train, y_train, X_test, y_test, hidden_size, epochs, batch_size, learning_rate):
    # Trainiert das Modell mit den angegebenen Hyperparametern.
    # Gibt die trainierten Parameter und die Loss/Accuracy-Verläufe zurück.
    
    # Parameter initialisieren
    W1, b1, W2, b2 = init_params(hidden_size=hidden_size)

    loss_history = []
    acc_history = []


     
    for epoch in range(epochs):
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)

        # Training in Mini-Batches
        for i in range(0, X_train.shape[0], batch_size):
            idx = indices[i:i+batch_size]
            X_batch = X_train[idx]

            # One-Hot Encoding für Labels im Batch
            y_batch = np.zeros((len(idx), 10))
            for j, label in enumerate(y_train[idx]):
                y_batch[j, label] = 1

            Z1, A1, Z2, A2 = forward_pass(X_batch, W1, b1, W2, b2)
            W1, b1, W2, b2 = backprop(X_batch, y_batch, Z1, A1, Z2, A2,
                                      W1, b1, W2, b2, learning_rate)

        # Nach jeder Epoche Loss und Accuracy berechnen
        loss = mse_calc(y_batch, A2)
        acc = evaluate_accuracy(X_test, y_test, W1, b1, W2, b2)

        loss_history.append(loss)
        acc_history.append(acc)

    return W1, b1, W2, b2, loss_history, acc_history

def grid_search(X_train, y_train, X_test, y_test,
                hidden_sizes, epoch_list, batch_sizes, learning_rate):
    # Führt eine Grid Search über verschiedene Hyperparameter-Kombinationen durch.
    # Testet alle vorgegebenen Kombinationen aus hidden_sizes, epoch_list und batch_sizes.
    # Gibt eine Liste der Ergebnisse zurück.
    
    results = []

    # Alle Kombinationen von Parametern testen
    for hidden_size in hidden_sizes:
        for epochs in epoch_list:
            for batch_size in batch_sizes:
                print(f"\nStarte Training mit hidden={hidden_size}, epochs={epochs}, batch={batch_size}")

                # Training mit aktueller Konfiguration
                W1, b1, W2, b2, loss_history, acc_history = train_model(
                    X_train, y_train, X_test, y_test,
                    hidden_size=hidden_size,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate
                )

                # Finale Accuracy (letzte Epoche)
                final_acc = acc_history[-1]

                # Ergebnis speichern
                results.append((hidden_size, epochs, batch_size, final_acc))

                print(f"Fertig! Test-Accuracy: {final_acc*100:.2f}%")

    return results

def plot_train_vs_test(train_acc_history, test_acc_history):
    # Vergleich aus Test- und Trainings Accuracy in Plot ausgeben
    epochs = range(1, len(train_acc_history) + 1)

    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_acc_history, label="Train-Accuracy", marker="o")
    plt.plot(epochs, test_acc_history, label="Test-Accuracy", marker="o")

    # Fokus auf den Bereich >85%, damit man die Unterschiede sieht
    plt.ylim(85, 100)

    plt.xlabel("Epoche")
    plt.ylabel("Accuracy (%)")
    plt.title("Train vs. Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main Programm
if __name__ == "__main__":
    data_dir = r"C:\Users\Willi\Desktop\Weiterbildung\AI Development\AI-Programme\PA_AIdev\mnist_data"  # Pfad, in dem .gz Dateien liegen

    # Daten vorbereiten:
    (X_train, y_train), (X_test, y_test) = load_mnist_dataset(data_dir)
    X_train, y_train = prepare_data(X_train, y_train)
    X_test, y_test = prepare_data(X_test, y_test)

    
    # Hyperparameter
    epochs = 30            # Netz wird x Mal trainiert
    batch_size = 32         # pro Epoche alle Bilder in Batches von y Bildern durchgehen

    # Initialisiere Gewichte und Biases
    W1, b1, W2, b2 = init_params()

    # für Plots
    train_acc_history = []
    test_acc_history = []

    # für early-Stopping: 
    best_acc = 0
    patience = 3   # wie viele Epochen warten ohne Verbesserung
    wait = 0       # Zähler für nicht verbesserte Epochen

    for epoch in range(epochs):
        # Indizes mischen
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)

        # Alle Batches in dieser Epoche durchlaufen
        for i in range(0, X_train.shape[0], batch_size):
            idx = indices[i:i+batch_size]
            X_batch = X_train[idx]

            # One-Hot-Encoding
            y_batch = np.zeros((len(idx), 10))
            for j, label in enumerate(y_train[idx]):
                y_batch[j, label] = 1

            # Forward Pass
            Z1, A1, Z2, A2 = forward_pass(X_batch, W1, b1, W2, b2)

            # Backpropagation
            W1, b1, W2, b2 = backprop(X_batch, y_batch, Z1, A1, Z2, A2,
                                    W1, b1, W2, b2, learning_rate)



        # --- Ende einer Epoche ---
        train_acc = evaluate_accuracy(X_train[:5000], y_train[:5000], W1, b1, W2, b2)  # nur Teilmenge für Geschwindigkeit
        test_acc = evaluate_accuracy(X_test, y_test, W1, b1, W2, b2)
        loss = mse_calc(y_batch, A2)

        train_acc_history.append(train_acc * 100)  # in %
        test_acc_history.append(test_acc * 100)

        print(f"Epoche {epoch+1}/{epochs} - Loss: {loss:.4f} - Train-Acc: {train_acc*100:.2f}% - Test-Acc: {test_acc*100:.2f}%")

        # --- Early Stopping prüfen ---
        if test_acc > best_acc:
            best_acc = test_acc
            wait = 0  # zurücksetzen, weil verbessert
        else:
            wait += 1
            if wait >= patience:
                print(f"-> Training gestoppt bei Epoche {epoch+1} (Early Stopping, beste Test-Acc={best_acc*100:.2f}%)")
                break
    
    # Plot ausgeben
    plot_train_vs_test(train_acc_history, test_acc_history) 


"""
    # ---------- Hyperparameter testen ---------------
    # läuft einmal sehr lange, daher auskommentiert. Ergebnis siehe Power-Point Präsentation


    hidden_sizes = [64, 128, 256]
    epoch_list = [10, 20, 30]
    batch_sizes = [32, 128, 512]
    learning_rate = 0.1

    results = grid_search(X_train, y_train, X_test, y_test,
                        hidden_sizes, epoch_list, batch_sizes,
                        learning_rate)

    # Ergebnisse aus deinem Lauf (Hidden, Epochen, Batch, Accuracy)
   
    # Ergebnisse sortieren (beste Accuracy zuerst)
    results.sort(key=lambda x: x[3], reverse=True)

    print("\n--- Ergebnisse ---")
    for hidden, epochs, batch, acc in results:
        print(f"Hidden={hidden}, Epochen={epochs}, Batch={batch}  Test-Accuracy: {acc*100:.2f}%")
    # -------------------------------------------------
"""