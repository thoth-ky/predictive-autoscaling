1. Create an Exection Role with necessary permissions for SageMaker Studio.

```bash
aws iam create-role --role-name execution-role-name --assume-role-policy-document file://execution-role-trust-policy.json
aws iam attach-role-policy --role-name execution-role-name --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
```
2. Get VPC and Subnet IDs.

```bash
aws --region region ec2 describe-vpcs --filters Name=isDefault,Values=true --query "Vpcs[0].VpcId" --output text
aws --region region ec2 describe-subnets --filters Name=vpc-id,Values=default-vpc-id --query "Subnets[*].SubnetId" --output json
```

3. Create Domain passing above details:

```bash
aws --region region sagemaker create-domain --domain-name domain-name --vpc-id default-vpc-id --subnet-ids subnet-ids --auth-mode authentication-mode \
    --default-user-settings "ExecutionRole=arn:aws:iam::account-number:role/execution-role-name,JupyterServerAppSettings={DefaultResourceSpec={InstanceType=system,SageMakerImageArn=image-arn}}" \
--query DomainArn --output text


4. List domains
```bash
aws --region region sagemaker list-domains
```
