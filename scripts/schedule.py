import os
import sys
from google.cloud import aiplatform

# Add the current directory to the path
sys.path.append(".")


# Cron schedules
cron_dict = {
    "every_minute": "* * * * *",
    "every_five_minutes": "*/5 * * * *",
    "every_ten_minutes": "*/10 * * * *",
    "every_fifteen_minutes": "*/15 * * * *",
    "every_half_hour": "*/30 * * * *",
    "every_hour": "0 * * * *",
    "every_two_hours": "0 */2 * * *",
    "every_three_hours": "0 */3 * * *",
    "every_six_hours": "0 */6 * * *",
    "every_twelve_hours": "0 */12 * * *",
    "every_day_midnight": "0 0 * * *",
    "every_day_noon": "0 12 * * *",
    "every_monday": "0 0 * * MON",
    "every_tuesday": "0 0 * * TUE",
    "every_wednesday": "0 0 * * WED",
    "every_thursday": "0 0 * * THU",
    "every_friday": "0 0 * * FRI",
    "every_saturday": "0 0 * * SAT",
    "every_sunday": "0 0 * * SUN",
    "every_weekend": "0 0 * * SAT,SUN",
    "every_weekday": "0 0 * * MON-FRI",
    "every_month_first_day": "0 0 1 * *",
    "every_month_last_day": "0 0 L * *",
    "every_year": "0 0 1 1 *",
}


# Constants
MODEL_NAME = "berkamodel"
PIPELINE_NAME = "production"
CRON_SCHEDULE = cron_dict["every_hour"]
MAX_CONCURRENT_RUN_COUNT = 1
MAX_RUN_COUNT = 10

# Environment variables
PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")
REPOSITORY = os.getenv("BUCKET_NAME")  # Match the Bucket name on Artifact Registry
BUCKET_NAME = os.getenv("BUCKET_NAME")

# Derived constants
BASE_IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPOSITORY}/{PIPELINE_NAME}:latest"
COMPILED_PIPELINE_PATH = f"gs://{BUCKET_NAME}/{PIPELINE_NAME}/artifacts/pipeline/{PIPELINE_NAME}_pipeline.json"
PIPELINE_ROOT_PATH = f"gs://{BUCKET_NAME}/{PIPELINE_NAME}/run/"
DISPLAY_NAME = f"{PIPELINE_NAME} Pipeline Job"
SCHEDULE_NAME = f"{PIPELINE_NAME} Schedule"


def main():
    # Initialize the AI Platform SDK
    aiplatform.init(project=PROJECT_ID, location=REGION)

    # Create a pipeline job
    pipeline_job = aiplatform.PipelineJob(
        display_name=DISPLAY_NAME,
        template_path=COMPILED_PIPELINE_PATH,
        pipeline_root=PIPELINE_ROOT_PATH,
    )

    # Create a schedule for the pipeline job
    pipeline_job.create_schedule(
        display_name=SCHEDULE_NAME,
        cron=CRON_SCHEDULE,
        max_concurrent_run_count=MAX_CONCURRENT_RUN_COUNT,
        max_run_count=MAX_RUN_COUNT,
    )

    # Run the pipeline job
    pipeline_job.run()


if __name__ == "__main__":
    main()
