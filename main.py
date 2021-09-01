from predict import predict_text

if __name__ == "__main__":
    text = "legitimate military targets, an"

    topic = predict_text(text, "naive_bayes_classifier.pkl")
    print(text)
    print(f"The topic is: {topic}")
