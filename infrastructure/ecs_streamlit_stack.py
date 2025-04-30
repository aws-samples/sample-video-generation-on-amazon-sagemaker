import ast
import os
from aws_cdk import (
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_ecs_patterns as ecs_patterns,
    aws_s3 as s3,
    aws_iam as iam,
    aws_ecr_assets as ecr_assets,
    aws_s3_deployment as s3_deployment,
    aws_logs as logs,
    aws_kms as kms,
    aws_secretsmanager as secretsmanager,
    aws_elasticloadbalancingv2 as elbv2,
    SecretValue,
    CfnOutput,
    RemovalPolicy,
    Duration,
    Stack,
)
from constructs import Construct
from cdk_nag import (
    NagSuppressions,
    NagPackSuppression,
)


class StreamlitEcsStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        NagSuppressions.add_stack_suppressions(self, [
            NagPackSuppression(
                id='AwsSolutions-IAM4',
                reason='AWS Lambda requires AWSLambdaBasicExecutionRole for CloudWatch Logs'
            ),
            NagPackSuppression(
                id='AwsSolutions-L1',
                reason='Custom resource Lambda functions are managed by CDK'
            ),
            NagPackSuppression(
                id='AwsSolutions-IAM5',
                reason='S3 and KMS actions require wildcard permissions for proper functionality',

            ),
            NagPackSuppression(
                id='AwsSolutions-SMG4',
                reason='Secret rotation not required for this use case'
            ),
            NagPackSuppression(
                id='AwsSolutions-ECS2',
                reason='Environment variables are required for application configuration'
            ),
            NagPackSuppression(
                id='AwsSolutions-EC23',
                reason='Application Load Balancer and Service need to be publicly accessible'
            ),
        ])

        docker_context_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        print(f"Docker context path: {docker_context_path}")

        shared_key = kms.Key(
            self, "SharedKey",
            enable_key_rotation=True,
            removal_policy=RemovalPolicy.DESTROY,
            alias="alias/shared-encryption-key",
            description="Shared KMS key for all encrypted resources"
        )

        vpc = ec2.Vpc(
            self, "StreamlitVpc",
            max_azs=2,
            nat_gateways=1,
            subnet_configuration=[
              ec2.SubnetConfiguration(
                  name="Private",
                  subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                  cidr_mask=24
              ),
              ec2.SubnetConfiguration(
                  name="Public",
                  subnet_type=ec2.SubnetType.PUBLIC,
                  cidr_mask=24
              )
            ]
        )

        flow_log_group = logs.LogGroup(
            self, "VPCFlowLogs",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
            encryption_key=shared_key
        )

        vpc.add_flow_log(
            "FlowLog",
            destination=ec2.FlowLogDestination.to_cloud_watch_logs(flow_log_group)
        )

        security_group = ec2.SecurityGroup(
            self, "ServiceSecurityGroup",
            vpc=vpc,
            description="Security group for Streamlit service",
            allow_all_outbound=False
        )

        security_group.add_ingress_rule(
            ec2.Peer.any_ipv4(),
            ec2.Port.tcp(8501),
            "Allow Streamlit inbound traffic"
        )

        security_group.add_egress_rule(
            ec2.Peer.any_ipv4(),
            ec2.Port.tcp(443),
            "Allow HTTPS outbound traffic"
        )

        cluster = ecs.Cluster(
            self, "StreamlitCluster",
            vpc=vpc, container_insights=True
        )

        shared_key.add_to_resource_policy(
            iam.PolicyStatement(
                sid="Allow S3 to use the key",
                actions=[
                    "kms:Decrypt",
                    "kms:GenerateDataKey"
                ],
                principals=[iam.ServicePrincipal("s3.amazonaws.com")],
                resources=["*"]
            )
        )

        shared_key.add_to_resource_policy(
            iam.PolicyStatement(
                sid="Allow Secrets Manager to use the key",
                actions=[
                    "kms:Decrypt",
                    "kms:Encrypt",
                    "kms:GenerateDataKey"
                ],
                principals=[iam.ServicePrincipal("secretsmanager.amazonaws.com")],
                resources=["*"]
            )
        )

        shared_key.add_to_resource_policy(
            iam.PolicyStatement(
                sid="Allow CloudWatch Logs to use the key",
                actions=[
                    "kms:Encrypt*",
                    "kms:Decrypt*",
                    "kms:ReEncrypt*",
                    "kms:GenerateDataKey*",
                    "kms:Describe*"
                ],
                principals=[iam.ServicePrincipal("logs.amazonaws.com")],
                resources=["*"]
            )
        )

        logs_bucket = s3.Bucket(
            self, "ALBLogsBucket",
            bucket_name=f"{self.account}-{self.region}-{construct_id}-logs".lower(),
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True
        )

        input_bucket = s3.Bucket(
            self, "VideoInputBucket",
            bucket_name=f"{self.account}-{self.region}-{construct_id}-video-input".lower(),
            encryption=s3.BucketEncryption.KMS,
            bucket_key_enabled=True,
            enforce_ssl=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            versioned=True,
            encryption_key=shared_key,
            server_access_logs_bucket=logs_bucket,
            server_access_logs_prefix="input-bucket-logs/",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            lifecycle_rules=[
                s3.LifecycleRule(
                 expiration=Duration.days(90),
                 noncurrent_version_expiration=Duration.days(7)
                )
            ]
        )

        output_bucket = s3.Bucket(
            self, "VideoOutputBucket",
            bucket_name=f"{self.account}-{self.region}-{construct_id}-video-output".lower(),
            encryption=s3.BucketEncryption.KMS,
            bucket_key_enabled=True,
            enforce_ssl=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            versioned=True,
            encryption_key=shared_key,
            server_access_logs_bucket=logs_bucket,
            server_access_logs_prefix="output-bucket-logs/",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            lifecycle_rules=[
                s3.LifecycleRule(
                 expiration=Duration.days(90),
                 noncurrent_version_expiration=Duration.days(7)
                )
            ]
        )

        process_code_dir = os.path.join(os.path.dirname(__file__), "processing_job")  
        processing_script_deployment = s3_deployment.BucketDeployment(
            self, "DeployProcessingJobCode",
            sources=[s3_deployment.Source.asset(process_code_dir)],
            destination_bucket=input_bucket,
            destination_key_prefix="scripts/"  
        )

        sagemaker_role = iam.Role(
            self, "SageMakerRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com")
        )

        sagemaker_role.add_to_policy(iam.PolicyStatement(
            actions=[
                "sagemaker:CreateProcessingJob",
                "sagemaker:DescribeProcessingJob",
            ],
            resources=[f"arn:aws:sagemaker:{Stack.of(self).region}:{Stack.of(self).account}:processing-job/*"]
        ))

        sagemaker_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "cloudwatch:GetMetricData",
                    "cloudwatch:ListMetrics"
                ],
                resources=[
                    "*"
                ]
            )
        )

        sagemaker_role.add_to_policy(iam.PolicyStatement(
            actions=[
                "ec2:CreateNetworkInterface",
                "ec2:CreateNetworkInterfacePermission",
                "ec2:DeleteNetworkInterface",
                "ec2:DeleteNetworkInterfacePermission",
                "ec2:DescribeNetworkInterfaces",
                "ec2:DescribeVpcs",
                "ec2:DescribeDhcpOptions",
                "ec2:DescribeSubnets",
                "ec2:DescribeSecurityGroups"
            ],
            resources=["*"]
        ))

        sagemaker_role.add_to_policy(iam.PolicyStatement(
            actions=[
                "sagemaker:CreateProcessingJob",
                "sagemaker:DescribeProcessingJob",
            ],
            resources=[f"arn:aws:sagemaker:{Stack.of(self).region}:{Stack.of(self).account}:processing-job/*"]
        ))

        sagemaker_role.add_to_policy(iam.PolicyStatement(
            actions=[
                "logs:CreateLogStream",
                "logs:PutLogEvents",
                "logs:CreateLogGroup",
                "logs:DescribeLogStreams"
            ],
            resources=[
                f"arn:aws:logs:{Stack.of(self).region}:{Stack.of(self).account}:log-group:/aws/sagemaker/ProcessingJobs:*",
                f"arn:aws:logs:{Stack.of(self).region}:{Stack.of(self).account}:log-group:/aws/sagemaker/ProcessingJobs:*:log-stream:*"
            ]
        ))

        sagemaker_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "cloudwatch:PutMetricData"
                ],
                resources=["*"],
                conditions={
                    "StringEquals": {
                        "cloudwatch:namespace": "/aws/sagemaker/ProcessingJobs"
                    }
                }
            )
        )

        input_bucket.grant_read(sagemaker_role)
        output_bucket.grant_write(sagemaker_role)

        task_role = iam.Role(
            self, "FargateTaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com")
        )

        input_bucket.grant_read_write(task_role)
        output_bucket.grant_read_write(task_role)
        
        task_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream"
                ],
                resources=[
                    "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-5-sonnet-20240620-v1:0"
                ]
            )
        )
        task_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "sagemaker:CreateProcessingJob",
                    "sagemaker:DescribeProcessingJob"
                ],
                resources=["arn:aws:sagemaker:*:*:processing-job/*"]
            )
        )
        task_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "logs:GetLogEvents",
                    "logs:DescribeLogStreams",
                    "logs:DescribeLogGroups"
                ],
                resources=[
                    "arn:aws:logs:*:*:log-group:/aws/sagemaker/ProcessingJobs:*",
                    "arn:aws:logs:*:*:log-group:/aws/sagemaker/ProcessingJobs:*:log-stream:*",
                    "arn:aws:logs:*:*:log-group::log-stream",
                    "*"
                ]
            )
        )
        task_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "cloudwatch:GetMetricData"
                ],
                resources=["*"]
            )
        )

        task_role.add_to_policy(iam.PolicyStatement(
            actions=["iam:PassRole"],
            resources=[sagemaker_role.role_arn]
        ))

        shared_key.grant_encrypt_decrypt(task_role)
        shared_key.grant_encrypt_decrypt(sagemaker_role)

        docker_image = ecr_assets.DockerImageAsset(
            self, "DockerImage",
            directory=docker_context_path,
            file="Dockerfile",
            asset_name="streamlit-app",
            platform=ecr_assets.Platform.LINUX_AMD64,
            build_args={
                "DOCKER_BUILDKIT": "1"
            },
        )

        env_secret = secretsmanager.Secret(
            self, "AppSecrets",
            secret_object_value={
                "SAGEMAKER_ROLE_ARN": SecretValue.unsafe_plain_text(sagemaker_role.role_arn),
            },
            encryption_key=shared_key,
        )

        private_subnets = vpc.select_subnets(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS).subnet_ids
        subnet_ids_string = ','.join(private_subnets)

        service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self, "Service",
            cluster=cluster,
            cpu=512,
            memory_limit_mib=1024,
            task_image_options=ecs_patterns.ApplicationLoadBalancedTaskImageOptions(
                image=ecs.ContainerImage.from_docker_image_asset(docker_image),
                container_port=8501,
                task_role=task_role,
                environment={
                    "INPUT_BUCKET_NAME": input_bucket.bucket_name,
                    "OUTPUT_BUCKET_NAME": output_bucket.bucket_name,
                    "PROCESSING_JOB_VPC_SUBNETS": subnet_ids_string,
                    "PROCESSING_JOB_SECURITY_GROUP_ID": security_group.security_group_id,
                },
                secrets={
                    "SAGEMAKER_ROLE_ARN": ecs.Secret.from_secrets_manager(env_secret, "SAGEMAKER_ROLE_ARN"),
                },
                enable_logging=True,
            ),
            public_load_balancer=True,
            platform_version=ecs.FargatePlatformVersion.LATEST,
            circuit_breaker={"rollback": True}
        )

        service.load_balancer.add_security_group(security_group)
        service.load_balancer.log_access_logs(logs_bucket)

        listener = service.listener
        default_action = elbv2.ListenerAction.fixed_response(
            status_code=403, content_type="text/plain", message_body="Access denied"
        )
        listener.add_action("default-action", action=default_action)

        ip_allow_ranges = ast.literal_eval(self.node.try_get_context('allowed_ips'))

        if not ip_allow_ranges:
            raise ValueError("No IP address provided. Please specify 'allowed_ips' in the context.")

        if isinstance(ip_allow_ranges, str):
            ip_allow_ranges = [ip_allow_ranges]

        if "0.0.0.0/0" in ip_allow_ranges:
            raise ValueError("Wildcard IP (0.0.0.0/0) is not allowed. Please specify a specific IP address.")

        # Add the allow rule for specified IP addresses
        listener.add_action(
            "AllowFromSpecificIPs",
            action=elbv2.ListenerAction.forward([service.target_group]),
            conditions=[
                elbv2.ListenerCondition.source_ips(ip_allow_ranges)
            ],
            priority=1
        )

        CfnOutput(self, "StreamlitURL", value=f"http://{service.load_balancer.load_balancer_dns_name}")
        CfnOutput(self, "InputBucketName", value=input_bucket.bucket_name)
        CfnOutput(self, "OutputBucketName", value=output_bucket.bucket_name)

