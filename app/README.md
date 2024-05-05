# Viva API

Author: InfinityLabs Team

This API provides an interface to predict the number of passengers that will board a flight in the future. The API uses a pre-trained model to predict the number of passengers that will board a flight in the future. The model is trained on the dataset provided by Viva and uses a prophet model to predict the number of passengers that will board a flight in the future.


## Table of Contents

- [Setup](#setup)
- [Installation](#installation)
- [Usage](#usage)

## Setup
1. Ensure you have Docker installed on your system.

## Installation

1. Download and extract the folder.
2. Navigate to the directory with the folder and build with 'docker build -t flights-prophet-app .'
3. Run with 'docker run -p 8080:8080 flights-prophet-app' to deploy.

## Usage

1. Open the link that appears on your Docker logs, or alternatively open your browser and navigate to http://localhost:8080.
2. Add '/docs' at the end of the link.
3. You can now use the web interface and upload the images inside the /images folder.
