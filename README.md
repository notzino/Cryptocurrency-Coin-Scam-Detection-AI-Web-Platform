# Crypto Prediction Platform

This platform allows you to predict whether a given cryptocurrency is a scam or legitimate. It does so by collecting and analyzing data from various sources using machine learning models.

This project does not include an executable (.exe) file. It is a web-based platform that needs to be set up and run by following the provided instructions. Once the steps are completed, the platform can be accessed via a web browser at the specified address.


# Setup Instructions

## 1. Download Redis
- [Download Redis](https://redis.io/downloads/) and place it in the main folder.

## 2. Download WordNet
- Ensure that WordNet is downloaded in the main folder. The script should automatically download it, but if it fails, download it manually.

## 3. Create and Activate a Virtual Environment
- Create a virtual environment:
  ```bash
  python -m venv venv
  ```
- Activate the virtual environment:
  - On Windows:
    ```bash
    .\venv\Scripts\activate
    ```
  - On macOS/Linux:
    ```bash
    source venv/bin/activate
    ```

## 4. Install Required Packages
- Install the required packages from the `requirements.txt` file:
  ```bash
  pip install -r requirements.txt
  ```

## 5. Setup MySQL Database
- Ensure MySQL is installed and running.
- Navigate to the `database_sql` folder and execute the SQL script to set up the database:
  ```bash
  mysql -u [your-username] -p [your-database-name] < database.sql
  ```
  Alternatively, you can manually copy, paste, and execute the SQL code.

## 6. Configure Environment Variables
- Open the `.env` file and update the following fields with your database information:
  ```plaintext
  DB_USER=[your-database-username]
  DB_PASSWORD=[your-database-password]
  DB_HOST=[your-database-host]
  DB_PORT=[your-database-port]
  DB_NAME=[your-database-name]
  ```
- Insert a `SECRET_KEY` in the `.env` file. You can generate one using the `secret.py` script provided:
  ```bash
  python secret.py
  ```

## 7. Insert API Keys in Config Files
- Open the config files and replace the existing API keys with your own. Note: The provided API keys are for testing purposes only.

## 8. Insert Initial Data
- Run the `insert_database_api.py` file to collect and insert the initial cryptocurrency data:
  ```bash
  python insert_database_api.py
  ```
- Wait until the script completes its execution.

## 9. Train the Model
- Run the `train_model.py` file to train the model:
  ```bash
  python models/train_model.py
  ```

## 10. Start the Platform
- Start the Redis broker server:
  ```bash
  cd redis
  .\redis-server.exe
  ```
- Start the Dramatiq task to handle asynchronous jobs and wait for it to be ready:
  ```bash
  dramatiq models.predict_new_coin --path .
  ```
- Run the platform:
  ```bash
  python app.py
  ```
- The platform should now be accessible at [http://127.0.0.1:8000/](http://127.0.0.1:8000/).

## 11. Deployment Notes
- If you deploy the system on a backend server, make sure to update the address in the JavaScript file (line 17) to match your server’s IP or domain:
  ```javascript
    const socket = io('http://[your-server-address]:8000/');
  ```
```
