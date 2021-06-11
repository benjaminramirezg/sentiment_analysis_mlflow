import pickle
import mlflow
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import text
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout

from mlflow import log_metric, log_param, log_artifacts

CONDA_ENV_PATH = './conda.yaml'
DATASET_PATH = './data/dataset.csv'
MODEL_PATH = 'model'
LOCAL_PATH = './model/'
THRESHOLD = 0.5

HYPERPARAMETERS = {
    'NUM_WORDS': 10000,
    'BATCH_SIZE': 16,
    'EPOCHS': 2,
    'DROPOUT': 0.2,
    'LEARNING_RATE': 0.001,
}

class ModelWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        import tensorflow as tf
        model = tf.keras.models.load_model(context.artifacts["model"])
        preprocessor = pickle.load(open(context.artifacts["preprocessor"],'rb'))
        self.model = model
        self.preprocessor = preprocessor

    def predict(self, context, model_input):
        texts = model_input['text'].tolist()
        X = self.preprocessor.texts_to_matrix(texts, mode='tfidf')
        p = self.model.predict(X)
        p = [1 if x[0] > THRESHOLD else 0 for x in p]
        return p


def train():
    """Trains a model given a dataset"""
    NUM_WORDS = HYPERPARAMETERS['NUM_WORDS']
    BATCH_SIZE = HYPERPARAMETERS['BATCH_SIZE']
    EPOCHS = HYPERPARAMETERS['EPOCHS']

    dataset = pd.read_csv(DATASET_PATH, sep='\t')
    texts = dataset["text"].tolist()
    labels = dataset["label"].tolist()

    tokenizer = text.Tokenizer(num_words=NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    X = tokenizer.texts_to_matrix(texts, mode="tfidf")
    y = np.array(labels)

    model = Sequential()
    model.add(Dense(250, input_shape=(NUM_WORDS,)))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()
    model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.25, verbose=2)
       
    evaluation = model.evaluate(X, y)
    metrics = {'loss': float(evaluation[0]), 'accuracy': float(evaluation[1])}

    nn_path = '{}/model.h5'.format(LOCAL_PATH)
    preprocessor_path = '{}/tokenizer.pkl'.format(LOCAL_PATH)
    model.save(nn_path)
    pickle.dump(tokenizer, open(preprocessor_path,'wb+'), protocol=pickle.HIGHEST_PROTOCOL)
    return metrics

if __name__ == "__main__":
    with mlflow.start_run():
        for key, value in HYPERPARAMETERS.items():
            mlflow.log_param(key, value)

        metrics = train()

        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        print('Ahora va...')
        mlflow.pyfunc.log_model(
            MODEL_PATH, loader_module='mlflow.pyfunc.PythonModel', data_path=LOCAL_PATH, conda_env=CONDA_ENV_PATH
            )
