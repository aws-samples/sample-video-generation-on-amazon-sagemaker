import streamlit as st
import os
from langchain_core.prompts import PromptTemplate
from langchain_aws import ChatBedrock
import uuid
import boto3
from datetime import datetime, timedelta
import time
import atexit

s3_client = boto3.client('s3')
sagemaker_client = boto3.client('sagemaker')
cloudwatch = boto3.client('cloudwatch')


def main():
    """
    Main app function
    """
    temp_dir = os.environ.get('TEMP_DIR', '/app/temp')
    os.makedirs(temp_dir, exist_ok=True)

    # Register cleanup function
    atexit.register(cleanup_temp_files)

    config_streamlit_app()
    init_session_variables()

    with st.sidebar:
        outputs = display_sidebar()

    st.header('Video Generation')

    display_outputs(outputs)

def config_streamlit_app():
    """
    Sets page settings / styles
    """
    st.set_page_config(
        page_title="Text to Video",
        page_icon="💡",
        layout="wide",
        initial_sidebar_state="expanded"
    )


# Initialise session state variables
def init_session_variables():
    initial_values = {
        'video_generated': False,
    }
    for k,v in initial_values.items():
        if k not in st.session_state:
            st.session_state[k] = v

    cleanup_temp_files()


def handle_user_prompt():
    user_prompt = st.sidebar.text_area("Enter your prompt")

    if st.sidebar.button("Enhance Prompt"):
        if user_prompt:
            st.session_state.user_prompt = user_prompt
            st.session_state.final_prompt = enhance_prompt(user_prompt)
        else:
            st.warning("Please write a prompt first.")
            st.session_state.user_prompt = None


def handle_generation():
    if st.session_state.final_prompt is not None:
        enhanced_prompt = st.sidebar.text_area(
            "Enhanced prompt, make any edits you need",
            value=st.session_state.final_prompt
        )
    else:
        enhanced_prompt = st.sidebar.text_area(
            "Enhanced prompt, make any edits you need",
            value=""
        )

    if st.session_state.uploaded_image is not None:
        uploaded_image = st.session_state.uploaded_image
    else:
        uploaded_image = None

    if st.sidebar.button("Generate video"):
        if enhanced_prompt and uploaded_image:
            st.session_state.final_prompt = enhanced_prompt
            st.session_state.video = generate_video_with_image(enhanced_prompt, uploaded_image)
            st.session_state.video_generated = True
        elif enhanced_prompt:
            st.session_state.final_prompt = enhanced_prompt
            # Set the video_generated flag to True
            st.session_state.video = generate_video(enhanced_prompt)
            st.session_state.video_generated = True
        else:
            st.warning("Please write a prompt first.")
            st.session_state.final_prompt = None
            st.session_state.video_generated = False


def enhance_prompt(original_prompt):
    model_id = 'anthropic.claude-3-5-sonnet-20240620-v1:0'
    model_kwargs = {
        "max_tokens": 4096,
        "temperature": 0.1
    }
    model = ChatBedrock(
        model_id=model_id,
        model_kwargs=model_kwargs,
        region_name = 'us-east-1'
    )

    prompt = '''
    <Role>
    Your role is to enhance the user prompt that is given to you by providing additional details to the prompt. The end goal is to
    covert the user prompt into a short video clip, so it is necessary to provide as much information you can.
    </Role>
    <Task>
    You must add details to the user prompt in order to enhance it for video generation. You must provide a 1 paragraph response. No more and no less.
    Only include the enhanced prompt in your response. Do not include anything else.
    </Task>
    <Prompt>
    {prompt}
    </Prompt>
    '''

    prompt = PromptTemplate(
        template = prompt,
        input_variables=["prompt"]
    )

    chain = prompt | model

    output = chain.invoke({"prompt": original_prompt})

    return output.content


def get_job_status(job_name):
    response = sagemaker_client.describe_processing_job(
        ProcessingJobName=job_name
    )
    return response['ProcessingJobStatus'], response.get('FailureReason', '')


def get_processing_job_progress(job_name):
    try:
        response = cloudwatch.get_metric_data(
            MetricDataQueries=[
                {
                    'Id': 'progress',
                    'MetricStat': {
                        'Metric': {
                            'Namespace': '/aws/sagemaker/ProcessingJobs',
                            'MetricName': 'ProgressPercentage',
                            'Dimensions': [
                                {
                                    'Name': 'ProcessingJobName',
                                    'Value': job_name
                                }
                            ]
                        },
                        'Period': 60,
                        'Stat': 'Maximum'
                    },
                    'ReturnData': True
                }
            ],
            StartTime=datetime.utcnow() - timedelta(hours=1),
            EndTime=datetime.utcnow()
        )

        # Get the latest progress value
        values = response['MetricDataResults'][0]['Values']
        return max(values)
    except Exception as e:
        print(f"Error fetching progress: {str(e)}")
        return 0.0


def monitor_processing_job(job_name, output_bucket, video_uid):
    # Create a status placeholder
    status_container = st.empty()
    log_container = st.empty()
    progress_text = "Processing job is running. This may take up to an hour"
    progress_bar = st.progress(0, text=progress_text)
    time.sleep(30)
    progress_bar.progress(10)

    while True:
        status, failure_reason = get_job_status(job_name)
        progress = get_processing_job_progress(job_name)
        progress_bar.progress(progress, text=progress_text)

        if status == 'Completed':
            progress_bar.progress(100, text="Completed!")
            video_filename = f"output/{video_uid}/generated_video.mp4"

            video_data = s3_client.get_object(Bucket=output_bucket, Key=video_filename)['Body'].read()
            return video_data

        elif status in ['Failed', 'Stopped']:
            status_container.error(f"Processing job {status}. Reason: {failure_reason}")
            break
        time.sleep(10)


def generate_video(enhanced_prompt):
    # generate uid for tracking
    video_uid = str(uuid.uuid4())

    # get bucket names from cdk output
    input_bucket = os.environ.get('INPUT_BUCKET_NAME')
    output_bucket = os.environ.get('OUTPUT_BUCKET_NAME')

    # upload enhanced prompt to s3 input bucket
    prompt_filename = f"input/{video_uid}.txt"
    s3_client.put_object(Bucket=input_bucket, Key=prompt_filename, Body=enhanced_prompt)

    # create sagemaker processing job
    job_name = f"video-gen-{video_uid}"
    response = sagemaker_client.create_processing_job(
        ProcessingJobName=job_name,
        ProcessingResources={
            'ClusterConfig': {
                'InstanceCount': 1,
                'InstanceType': 'ml.g5.4xlarge',
                'VolumeSizeInGB': 200
            }
        },
        AppSpecification={
            'ImageUri': '763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker',
            'ContainerEntrypoint': [
                'bash', '-c',
                'pip install -r /opt/ml/processing/scripts/requirements.txt && python /opt/ml/processing/scripts/inference.py'
            ]
        },
        ProcessingInputs=[
            {
                'InputName': 'text_input',
                'S3Input': {
                    'S3Uri': f"s3://{input_bucket}/{prompt_filename}",
                    'LocalPath': '/opt/ml/processing/input/text/',
                    'S3DataType': 'S3Prefix',
                    'S3InputMode': 'File',
                }
            },
            {
                'InputName': 'scripts',
                'S3Input': {
                    'S3Uri': f"s3://{input_bucket}/scripts",
                    'LocalPath': '/opt/ml/processing/scripts',
                    'S3DataType': 'S3Prefix',
                    'S3InputMode': 'File',
                }
            }
        ],
        ProcessingOutputConfig={
            'Outputs': [
                {
                    'OutputName': 'mp4_output',
                    'S3Output': {
                        'S3Uri': f"s3://{output_bucket}/output/{video_uid}/",
                        'LocalPath': '/opt/ml/processing/output/',
                        'S3UploadMode': 'EndOfJob',
                    }
                }
            ]
        },
        Environment={
            'prompt_filename': f"{video_uid}.txt",
            'text_path': '/opt/ml/processing/input/text',
            'region': 'us-east-1',
            'job_name': job_name
        },
        RoleArn=os.environ.get('SAGEMAKER_ROLE_ARN'),
        StoppingCondition={'MaxRuntimeInSeconds': 7200},
        NetworkConfig={
            'EnableInterContainerTrafficEncryption': True,
            'EnableNetworkIsolation': False,
            'VpcConfig': {
                'SecurityGroupIds': [os.getenv('PROCESSING_JOB_SECURITY_GROUP_ID')],
                'Subnets': os.getenv('PROCESSING_JOB_VPC_SUBNETS').split(',')
            }
        }
    )

    monitor_processing_job(job_name, output_bucket, video_uid)
    return video_uid


def generate_video_with_image(enhanced_prompt, uploaded_image):
    video_uid = str(uuid.uuid4())

    input_bucket = os.environ.get('INPUT_BUCKET_NAME')
    output_bucket = os.environ.get('OUTPUT_BUCKET_NAME')

    prompt_filename = f"input/{video_uid}.txt"
    s3_client.put_object(Bucket=input_bucket, Key=prompt_filename, Body=enhanced_prompt)

    # Get file extension from the uploaded file name
    file_extension = os.path.splitext(uploaded_image.name)[1].lstrip('.')
    if not file_extension:
        file_extension = "jpg"  # Default extension if none is found
        
    image_filename = f"input/{video_uid}.{file_extension}"
    s3_client.put_object(Bucket=input_bucket, Key=image_filename, Body=uploaded_image.getvalue())

    job_name = f"video-gen-{video_uid}"
    response = sagemaker_client.create_processing_job(
        ProcessingJobName=job_name,
        ProcessingResources={
            'ClusterConfig': {
                'InstanceCount': 1,
                'InstanceType': 'ml.g5.4xlarge',
                'VolumeSizeInGB': 200
            }
        },
        AppSpecification={
            'ImageUri': '763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker',
            'ContainerEntrypoint': [
                'bash', '-c',
                'pip install -r /opt/ml/processing/scripts/requirements.txt && python /opt/ml/processing/scripts/inference.py'
            ]
        },
        ProcessingInputs=[
            {
                'InputName': 'text_input',
                'S3Input': {
                    'S3Uri': f"s3://{input_bucket}/{prompt_filename}",
                    'LocalPath': '/opt/ml/processing/input/text/',
                    'S3DataType': 'S3Prefix',
                    'S3InputMode': 'File',
                }
            },
            {
                'InputName': 'image_input',
                'S3Input': {
                    'S3Uri': f"s3://{input_bucket}/{image_filename}",
                    'LocalPath': '/opt/ml/processing/input/image/',
                    'S3DataType': 'S3Prefix',
                    'S3InputMode': 'File',
                }
            },
            {
                'InputName': 'scripts',
                'S3Input': {
                    'S3Uri': f"s3://{input_bucket}/scripts",
                    'LocalPath': '/opt/ml/processing/scripts',
                    'S3DataType': 'S3Prefix',
                    'S3InputMode': 'File',
                }
            }
        ],
        ProcessingOutputConfig={
            'Outputs': [
                {
                    'OutputName': 'mp4_output',
                    'S3Output': {
                        'S3Uri': f"s3://{output_bucket}/output/{video_uid}/",
                        'LocalPath': '/opt/ml/processing/output/',
                        'S3UploadMode': 'EndOfJob',
                    }
                }
            ]
        },
        Environment={
            'prompt_filename': f"{video_uid}.txt",
            'image_filename': f"{video_uid}.{file_extension}",
            'image_path': '/opt/ml/processing/input/image',
            'text_path': '/opt/ml/processing/input/text',
            'region': 'us-east-1',
            'job_name': job_name
        },
        RoleArn=os.environ.get('SAGEMAKER_ROLE_ARN'),
        StoppingCondition={'MaxRuntimeInSeconds': 7200},
        NetworkConfig={
            'EnableInterContainerTrafficEncryption': True,
            'EnableNetworkIsolation': False,
            'VpcConfig': {
                'SecurityGroupIds': [os.getenv('PROCESSING_JOB_SECURITY_GROUP_ID')],
                'Subnets': os.getenv('PROCESSING_JOB_VPC_SUBNETS').split(',')
            }
        }
    )

    monitor_processing_job(job_name, output_bucket, video_uid)
    return video_uid


def cleanup_temp_files():
    """Cleanup temporary files on exit"""
    try:
        temp_dir = os.environ.get('TEMP_DIR', '/app/temp')
        for filename in os.listdir(temp_dir):
            if filename.startswith('temp_') and filename.endswith('.mp4'):
                file_path = os.path.join(temp_dir, filename)
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error removing {file_path}: {str(e)}")
    except Exception as e:
        print(f"Error cleaning up temp directory: {str(e)}")


def display_video():
    if st.session_state.video_generated:
        try:
            video_uid = st.session_state.video
            output_bucket = os.environ.get('OUTPUT_BUCKET_NAME')
            video_key = f"output/{video_uid}/generated_video.mp4"

            temp_dir = os.environ.get('TEMP_DIR', '/app/temp')
            os.makedirs(temp_dir, exist_ok=True)

            temp_video_path = os.path.join(temp_dir, f"temp_{video_uid}.mp4")

            try:
                video_data = s3_client.get_object(
                    Bucket=output_bucket,
                    Key=video_key
                )['Body'].read()

                with open(temp_video_path, "wb") as f:
                    f.write(video_data)

                st.write("### Generated Video")
                st.video(temp_video_path)

                st.download_button(
                    label="Download Video",
                    data=video_data,
                    file_name=f"{video_uid}_generated_video.mp4",
                    mime="video/mp4"
                )

                try:
                    os.remove(temp_video_path)
                except Exception as e:
                    print(f"Error cleaning up temp file: {str(e)}")

            except Exception as e:
                st.error(f"Error loading video from S3: {str(e)}")

        except Exception as e:
            st.error(f"Error displaying video: {str(e)}")

def reset_session_state():
    """
    Clears all session state variables to reset the app.
    """
    for key in st.session_state.keys():
        st.session_state[key] = None

def display_sidebar():
    st.title("Video Generator")

    if "user_prompt" not in st.session_state:
        st.session_state.user_prompt = None

    if "final_prompt" not in st.session_state:
        st.session_state.final_prompt = None

    if "video_generated" not in st.session_state:
        st.session_state.video_generated = False

    if "video" not in st.session_state:
        st.session_state.video = None

    if "include_image" not in st.session_state:
        st.session_state.include_image = False

    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None

    handle_user_prompt()

    st.session_state.include_image = st.sidebar.checkbox("Include an image?", value=st.session_state.include_image)

    if st.session_state.include_image:
        with st.sidebar.expander("Upload Image"):
            uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                st.session_state.uploaded_image = uploaded_file
                st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    handle_generation()

    input_selection = {
        'user_prompt': st.session_state.user_prompt,
        'final_prompt': st.session_state.final_prompt,
        'include_image': st.session_state.include_image,
        'uploaded_image': st.session_state.uploaded_image
    }

    return {'input_selection': input_selection}


def display_outputs(outputs):
    display_video()


if __name__ == '__main__':
    main()
