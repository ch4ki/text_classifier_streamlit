import pickle

from utilities import clean_text


def predict_text(text, model_src):

    with open(model_src, 'rb') as fid:
        model = pickle.load(fid)

    labels = ['Business', 'Entertainment', 'Politics', 'Sport', 'Tech']

    clean_test = clean_text(text)
    prediction = model.predict([clean_test])[0]
    #print(f"Topic category is: {labels[prediction]}")

    return labels[prediction]