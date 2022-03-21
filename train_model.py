# Packages
import joblib

# Get customized functions from library
import packages.data_processor as dp
import packages.model_trainer as mt


# 0.Path to data
path_to_data = './data/spam.csv'

# 1.Prepare the data
prepared_data = dp.prepare_data(path_to_data, encoding="latin-1")

# 2.Create train - test split
train_test_data, vectorizer = dp.create_train_test_data(prepared_data['text'], 
                                         prepared_data['label'], 
                                         0.33, 2021)

# 3.Run training
model = mt.run_model_training(train_test_data['x_train'], train_test_data['x_test'], 
                           train_test_data['y_train'], train_test_data['y_test'])

# 4.Save the trained model and vectorizer
joblib.dump(model, './models/spam_detector_model.pkl')
joblib.dump(vectorizer, open("./vectors/vectorizer.pickle", "wb")) 
