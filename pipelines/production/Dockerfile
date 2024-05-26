# Multistage Dockerfile to build the production image
FROM europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest

# ARG PIPELINE_NAME to match the pipeline name in the pipelines directory
ARG PIPELINE_NAME
ARG PROJECT_ID
ARG REGION
ARG BUCKET_NAME

# Set the working directory to the pipeline name
WORKDIR /${PIPELINE_NAME}

# copy the pipeline code to the container
COPY pipelines/${PIPELINE_NAME}/ .

# Set ENV variables based on ARGS
ENV PROJECT_ID=$PROJECT_ID
ENV REGION=$REGION
ENV BUCKET_NAME=$BUCKET_NAME
# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
ENV GOOGLE_APPLICATION_CREDENTIALS=service-account.json
ENV FLASK_APP=./app/app.py

# Install the pipeline code
RUN pip install --upgrade pip 
RUN pip install -e .

# Expose port 8080
EXPOSE 8080
ENTRYPOINT ["flask", "run", "--host=0.0.0.0", "--port=8080"]



