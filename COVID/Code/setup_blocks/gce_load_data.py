import google.auth
from google.cloud import bigquery
from google.cloud import bigquery_storage_v1beta1



# Explicitly create a credentials object. This allows you to use the same
# credentials for both the BigQuery and BigQuery Storage clients, avoiding
# unnecessary API calls to fetch duplicate authentication tokens.
credentials, your_project_id = google.auth.default(
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

# Make clients.
bqclient = bigquery.Client(
    credentials=credentials,
    project=your_project_id,
)
bqstorageclient = bigquery_storage_v1beta1.BigQueryStorageClient(
    credentials=credentials
)

# Download query results.
query_string = """
SELECT *
FROM `bigquery-public-data.covid19_jhu_csse.confirmed_cases`
"""

diagnosed_ground_truth = (
    bqclient.query(query_string)
    .result()
    .to_dataframe(bqstorage_client=bqstorageclient)
)
diagnosed_ground_truth['type']='diagnosed'
query_string = """
SELECT *
FROM `bigquery-public-data.covid19_jhu_csse.deaths`
"""

fatality_ground_truth = (
    bqclient.query(query_string)
    .result()
    .to_dataframe(bqstorage_client=bqstorageclient)
)
fatality_ground_truth['type']='fatality'

ground_truth_global = diagnosed_ground_truth.append(fatality_ground_truth)

ground_truth_global.reset_index(drop=True, inplace =True) 
ground_truth_global.drop(index = ground_truth_global.dropna().index, inplace = True)

ground_truth_global.to_pickle("./.ground_truth_pd")




