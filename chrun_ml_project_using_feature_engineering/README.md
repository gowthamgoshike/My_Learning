
Customer Churn Prediction Project
================================

This project predicts customer churn for a telecom company using machine learning.  
It uses feature engineering, preprocessing, and a pipeline to train a model and make predictions.



How to Run
----------

1. Make sure Python 3.8+ is installed.
2. Install dependencies:
   pip install -r requirements.txt
3. Put your dataset CSV (Telco Customer Churn) in the project root or a folder of your choice.
4. If needed, update the dataset path in `main.py`.
5. Run the pipeline:
   python main.py
   - This will read the dataset, apply feature engineering and preprocessing, train the model, and print predictions.

Running Tests
-------------

- To run unit tests:
  python -m pytest
- Tests are located in the `tests/` folder and validate feature engineering and pipeline functionality.

Notes
-----

- Keep the dataset CSV as it is; the pipeline reads it directly.
- Missing numeric values are handled automatically in the pipeline.
- Logistic Regression is used by default, but you can replace it with other models if desired.
