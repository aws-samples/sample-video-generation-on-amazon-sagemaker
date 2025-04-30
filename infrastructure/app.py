#!/usr/bin/env python3
import os
import aws_cdk as cdk
from aws_cdk import Aspects
from cdk_nag import AwsSolutionsChecks, NagSuppressions
from ecs_streamlit_stack import StreamlitEcsStack

app = cdk.App()
stack = StreamlitEcsStack(app, "VideoGenerationStack",
    env=cdk.Environment(account=os.getenv('CDK_DEFAULT_ACCOUNT'), region=os.getenv('CDK_DEFAULT_REGION')),
)

Aspects.of(app).add(AwsSolutionsChecks())

app.synth()
