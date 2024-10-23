# Search Enhancement API

This API provides an endpoint that processes a user's long-form query, extracts key points, retrieves search results via the Google Search API, and returns the relevant results.

## Contributors

- Ruan Buhr (26440873)
- David Genders (25935410)
- Marcell Muller (23603097)

## Dataset Used

the dataset used to train the deep learning model can be found at: https://huggingface.co/datasets/sentence-transformers/stsb

## Getting Started

Follow these instructions to run the API server and make requests.

### Prerequisites

Ensure you have Python 3 installed on your machine.

### Installation

1. Clone the repository.
2. Navigate to the project directory.
3. Install the required dependencies using the following command:

   ```
   pip install -r requirements.txt
   ```

## Running the Server

To start the server, run the following command:

```
python3 main.py
```

## API Usage

Once the server is running you can make a POST request to the following endpoint:

```
POST http://localhost:5000/search
```

#### Request Headers

- Content-Type: application/json

#### Request Body

- Include your search query in the body of the request as JSON. For example:

```
{
    "query": "What is the Python programming language?"
}
```

## Response

The API will return a JSON response containing a list of links to pages relevant to the query.

## Using the Hosted Version

Alternatively you could use the hosted version of the search enhancement application hosted on Google Cloud Run at the following URL: https://search-enhancement-571422257357.africa-south1.run.app/search.

You can make a POST request to the above endpoint as follows:

```
POST https://search-enhancement-571422257357.africa-south1.run.app/search
```

#### Request Headers

- Content-Type: application/json

#### Request Body

- Include your search query in the body of the request as JSON. For example:

```
{
    "query": "What is the C programming language?"
}
```
