### To deploy a fitted scikit-learn model

- import packages
- connect to workspace
- deploy locally
  - define the endpoint
  - define the deployment (blue deployment)
    - model
    - scoring script
    - environments
    - instance type
  - create local endpoint  (local=True)
  - create local deployment (local=True) (using the blue deployment)
  - verify it is deployed
  - test the local deployment
- deploy online to azure
  - create the endpoint
  - create the deployment (this will deploy the endpoint)
  - test the online deployment
- Delete the endpoint


### Project lifecycle
![Project lifecycle](image-6.png)

### ML model lifecycle
![ML model lifecycle](image-7.png)