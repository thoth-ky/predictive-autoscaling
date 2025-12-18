resource "aws_iam_role" "aws_sagemaker_role" {
  name               = "aws_sagemaker_role"
  path               = "/"
  assume_role_policy = data.aws_iam_policy_document.aws_sagemaker_role.json
}

# Attach the SageMaker execution policy
resource "aws_iam_role_policy_attachment" "sagemaker_execution_policy" {
  role       = aws_iam_role.aws_sagemaker_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

# Add KMS permissions for SageMaker
resource "aws_iam_role_policy" "sagemaker_kms_policy" {
  name = "sagemaker_kms_policy"
  role = aws_iam_role.aws_sagemaker_role.id

  policy = data.aws_iam_policy_document.sagemaker_kms_policy.json
}

# Add EFS permissions for SageMaker
resource "aws_iam_role_policy" "sagemaker_efs_policy" {
  name = "sagemaker_efs_policy"
  role = aws_iam_role.aws_sagemaker_role.id

  policy = data.aws_iam_policy_document.sagemaker_efs_policy.json
}

data "aws_iam_policy_document" "sagemaker_efs_policy" {
  statement {
    effect = "Allow"
    actions = [
      "elasticfilesystem:CreateFileSystem",
      "elasticfilesystem:CreateMountTarget",
      "elasticfilesystem:CreateAccessPoint",
      "elasticfilesystem:DescribeFileSystems",
      "elasticfilesystem:DescribeMountTargets",
      "elasticfilesystem:DescribeAccessPoints",
      "elasticfilesystem:DeleteFileSystem",
      "elasticfilesystem:DeleteMountTarget",
      "elasticfilesystem:DeleteAccessPoint"
    ]
    resources = ["*"]
  }
}

data "aws_iam_policy_document" "sagemaker_kms_policy" {
  statement {
    effect = "Allow"
    actions = [
      "kms:CreateGrant",
      "kms:Decrypt",
      "kms:DescribeKey",
      "kms:Encrypt",
      "kms:GenerateDataKey",
      "kms:ReEncrypt*"
    ]
    resources = ["*"]
  }
}

data "aws_iam_policy_document" "aws_sagemaker_role" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}
