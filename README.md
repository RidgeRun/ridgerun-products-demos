# RidgeRun Products Demos and Reference Designs

In this repository, you can find demos, reference designs, and tests scripts of RidgeRun products, for you to easily evaluate them or take as a reference when creating your own applications.

If you want to evaluate RidgeRun products, don't hesitate to contact us for details at support@ridgerun.com

## Getting started
You will find two folders:
- `products-demos`: contains product-specific demos, scripts and examples to show the product functionality and features.
- `reference-designs`: contains reference designs, demos with combined products, and combined usage examples to show complex use cases. 

## Running Docker on x86
The Docker image for x86 has all the needed dependencies installed to run the RidgeRun products. Download the Docker image:
```
docker pull ridgerun/products-evals-demos:ridgerun/products-evals-demos
```
This container needs some additional permissions to run the demo scripts. To make running the container easier, the <code>compose.yaml</code> file has the settings needed to run the examples correctly. To start the docker run: 
```
docker compose run --name product-demo-container product-demos
```
Some of the examples need a second terminal to view results. A second terminal can be opened by running:
```
docker exec -it product-demo-container /bin/bash /product-demo/demo.sh
```
