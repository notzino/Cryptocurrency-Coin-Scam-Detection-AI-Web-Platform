import os
from dotenv import load_dotenv
import pymysql
from pymysql.cursors import DictCursor

load_dotenv()
class Database:
    def __init__(self):
        self.session = None
        self.connection = pymysql.connect(
            host=os.getenv('DB_HOST', ''),
            user=os.getenv('DB_USER', ''),
            password=os.getenv('DB_PASSWORD', ''),
            port=int(os.getenv('DB_PORT', 3306)),
            db=os.getenv('DB_NAME', ''),
            cursorclass=pymysql.cursors.DictCursor
        )
        self.cursor = self.connection.cursor()

    def execute(self, query, params=None):
        self.cursor.execute(query, params)
        self.connection.commit()

    def fetchone(self, query, params=None):
        self.cursor.execute(query, params)
        result = self.cursor.fetchone()
        self.clear_results()
        return result

    def fetchall(self, query, params=None):
        self.cursor.execute(query, params)
        result = self.cursor.fetchall()
        self.clear_results()
        return result

    def clear_results(self):
        while self.cursor.nextset():
            try:
                self.cursor.fetchall()
            except:
                pass

    def commit(self):
        self.connection.commit()

    def close(self):
        self.cursor.close()
        self.connection.close()
