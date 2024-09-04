-- Create the database
CREATE DATABASE IF NOT EXISTS crypto_db;

-- Use the database
USE crypto_db;

-- Table: cryptocurrencies
CREATE TABLE `cryptocurrencies` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `coin_name` VARCHAR(255) NOT NULL,
  `symbol` VARCHAR(50) NOT NULL,
  `is_scam` BOOLEAN NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `coin_unique` (`coin_name`, `symbol`)
);

-- Table: historical_data
CREATE TABLE `historical_data` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `coin_id` INT NOT NULL,
  `timestamp` DATETIME NOT NULL,
  `price` DECIMAL(20, 8) NOT NULL,
  `volume` DECIMAL(20, 8) NOT NULL,
  PRIMARY KEY (`id`),
  FOREIGN KEY (`coin_id`) REFERENCES `cryptocurrencies`(`id`)
);

-- Table: news_articles
CREATE TABLE `news_articles` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `coin_id` INT NOT NULL,
  `title` TEXT NOT NULL,
  `description` TEXT NOT NULL,
  `url` TEXT NOT NULL,
  `published_at` DATETIME NOT NULL,
  PRIMARY KEY (`id`),
  FOREIGN KEY (`coin_id`) REFERENCES `cryptocurrencies`(`id`)
);

-- Table: social_media
CREATE TABLE `social_media` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `coin_id` INT NOT NULL,
  `platform` VARCHAR(50) NOT NULL,
  `content` TEXT NOT NULL,
  `created_at` DATETIME NOT NULL,
  `num_comments` INT NOT NULL,
  `score` INT NOT NULL,
  PRIMARY KEY (`id`),
  FOREIGN KEY (`coin_id`) REFERENCES `cryptocurrencies`(`id`)
);

-- Table: wallet_data
CREATE TABLE `wallet_data` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `coin_id` INT NOT NULL,
  `address` TEXT NOT NULL,
  `transaction_data` JSON NOT NULL,
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  FOREIGN KEY (`coin_id`) REFERENCES `cryptocurrencies`(`id`)
);

-- Table: wallet_transactions
CREATE TABLE `wallet_transactions` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `wallet_data_id` INT NOT NULL,
  `block_id` TEXT NOT NULL,
  `transaction_hash` TEXT NOT NULL,
  `time` DATETIME NOT NULL,
  `balance_change` DECIMAL(20, 8) NOT NULL,
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  FOREIGN KEY (`wallet_data_id`) REFERENCES `wallet_data`(`id`)
);

-- Table: task_status
CREATE TABLE `task_status` (
  `session_id` VARCHAR(255) NOT NULL,
  `status` VARCHAR(50) NOT NULL,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`session_id`)
);

-- Table: predictions
CREATE TABLE `predictions` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `coin_id` INT NOT NULL,
  `symbol` VARCHAR(50) NOT NULL,
  `coin_name` VARCHAR(255) NOT NULL,
  `prediction` BOOLEAN NOT NULL,
  PRIMARY KEY (`id`),
  FOREIGN KEY (`coin_id`) REFERENCES `cryptocurrencies`(`id`)
);

-- Table: models
CREATE TABLE `models` (
  `model_id` INT NOT NULL AUTO_INCREMENT,
  `model_name` VARCHAR(255) NOT NULL,
  `model_file` BLOB NOT NULL,
  PRIMARY KEY (`model_id`)
);

-- Table: sessions
CREATE TABLE `sessions` (
  `session_id` VARCHAR(255) NOT NULL,
  `data` BLOB NOT NULL,
  expiration TIMESTAMP,
  PRIMARY KEY (`session_id`)
);

ALTER TABLE `sessions`
DROP PRIMARY KEY;

ALTER TABLE `sessions`
ADD COLUMN `id` INT AUTO_INCREMENT PRIMARY KEY FIRST,
ADD UNIQUE (`session_id`);

ALTER TABLE `sessions`
CHANGE COLUMN `expiration` `expiry` TIMESTAMP;

ALTER TABLE historical_data MODIFY COLUMN volume DECIMAL(30, 10) DEFAULT 0;
ALTER TABLE historical_data MODIFY COLUMN price DECIMAL(18, 8) DEFAULT 0;

ALTER TABLE wallet_transactions MODIFY COLUMN balance_change DECIMAL(30, 10);

