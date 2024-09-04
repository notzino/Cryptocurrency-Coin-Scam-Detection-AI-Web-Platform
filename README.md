# Crypto Prediction Platform

This platform allows you to predict whether a given cryptocurrency is a scam or legitimate. It does so by collecting and analyzing data from various sources using machine learning models.

This project does not include an executable (.exe) file. It is a web-based platform that needs to be set up and run by following the provided instructions. Once the steps are completed, the platform can be accessed via a web browser at the specified address.


# Setup Instructions

Setup Instructions

Follow these steps to set up and run the platform locally.

1. Download Redis

- Download Redis and place redis in the main folder

   https://redis.io/downloads/

2. Download WordNet

- Make sure Wordnet is downloaded in the main folder category (The script should download it automatically but in any case it fails download it manually)


3. Create and Activate a Virtual Environment

- Create a virtual environment:

  python -m venv venv

- Activate the virtual environment:

  - On Windows:
    .\venv\Scripts\activate
  - On macOS/Linux:
    source venv/bin/activate

4. Install Required Packages

- Install the required packages from the requirements.txt file:

  - pip install -r requirements.txt

5. Setup MySQL Database

- Ensure MySQL is installed and running.
- Navigate to the database_sql folder and execute the SQL script to set up the database:
- Or you can access the file and copy, paste and execute the sql code.

  mysql -u [your-username] -p [your-database-name] < database.sql

6. Configure Environment Variables

- Open the .env file and update the following fields with your database information:

  DB_USER=[your-database-username]
  DB_PASSWORD=[your-database-password]
  DB_HOST=[your-database-host]
  DB_PORT=[your-database-port]
  DB_NAME=[your-database-name]

- Also, insert a SECRET_KEY in the .env file. You can generate one using the secret.py script provided:

  python secret.py

7. Insert API Keys in Config Files

- Open the config files and replace the existing API keys with your own. Note: The provided API keys are for testing purposes.

8. Insert Initial Data

- Run the insert_database_api.py file to collect and insert the initial cryptocurrency data:

  python insert_database_api.py

  Wait until the script completes its execution.

9. Train the Model

- Run the train_model.py file to train the model:

  python models/train_model.py

10. Start the Platform

- To start the Redis broker server:

  cd redis
  .\redis-server.exe

- Start the dramatiq task to handle asynchronous jobs & wait for it to be ready:

  dramatiq models.predict_new_coin --path .

- Run the platform:

  python app.py

- The platform should now be accessible at http://127.0.0.1:8000/.

11. Deployment Notes

- If you deploy the system on a backend server, make sure to update the address in the JavaScript file (line 17) to match your server’s IP or domain:

  const socket = io('http://[your-server-address]:8000/');


