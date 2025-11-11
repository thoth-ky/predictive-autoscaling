variable "sagemaker" {
  description = "Configuration for SageMaker"
  type = object({
    domain_name            = string
    notebook_instance_name = string
    notebook_instance_type = string
    space_name             = string
    user_profile_name      = string
    gh_repo_url            = string
  })
}

resource "aws_sagemaker_domain" "sagemaker_domain" {
  domain_name = var.sagemaker.domain_name
  auth_mode   = "IAM" //  or "SSO" based on your requirement
  vpc_id      = aws_vpc.sagemaker_vpc.id
  subnet_ids  = [aws_subnet.sagemaker_subnet.id]

  # Use AWS managed key for EFS encryption
  kms_key_id = aws_kms_key.sagemaker_kms_key.key_id

  default_user_settings {
    execution_role = aws_iam_role.aws_sagemaker_role.arn
  }

  default_space_settings {
    execution_role = aws_iam_role.aws_sagemaker_role.arn
  }
}

# KMS key for SageMaker EFS encryption
resource "aws_kms_key" "sagemaker_kms_key" {
  description             = "KMS key for SageMaker domain EFS encryption"
  deletion_window_in_days = 7

  policy = data.aws_iam_policy_document.sagemaker_kms_key_policy.json

  tags = {
    Name = "sagemaker-efs-key"
  }
}

# KMS key policy to allow SageMaker and EFS services
data "aws_iam_policy_document" "sagemaker_kms_key_policy" {
  # Allow root account to manage the key
  statement {
    sid    = "Enable IAM policies"
    effect = "Allow"
    principals {
      type        = "AWS"
      identifiers = ["arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"]
    }
    actions   = ["kms:*"]
    resources = ["*"]
  }

  # Allow SageMaker service to use the key
  statement {
    sid    = "Allow SageMaker Service"
    effect = "Allow"
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
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

  # Allow EFS service to use the key
  statement {
    sid    = "Allow EFS Service"
    effect = "Allow"
    principals {
      type        = "Service"
      identifiers = ["elasticfilesystem.amazonaws.com"]
    }
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

  # Allow SageMaker role to use the key
  statement {
    sid    = "Allow SageMaker Role"
    effect = "Allow"
    principals {
      type        = "AWS"
      identifiers = [aws_iam_role.aws_sagemaker_role.arn]
    }
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

# Data source to get current AWS account ID
data "aws_caller_identity" "current" {}

resource "aws_kms_alias" "sagemaker_kms_alias" {
  name          = "alias/sagemaker-efs-key"
  target_key_id = aws_kms_key.sagemaker_kms_key.key_id
}

resource "aws_sagemaker_user_profile" "default_sagemaker_user_profile" {
  domain_id         = aws_sagemaker_domain.sagemaker_domain.id
  user_profile_name = var.sagemaker.user_profile_name
}

resource "aws_sagemaker_space" "shared_space" {
  domain_id  = aws_sagemaker_domain.sagemaker_domain.id
  space_name = var.sagemaker.space_name
}


resource "aws_sagemaker_code_repository" "gh_repo" {
  code_repository_name = "code-repo"

  git_config {
    repository_url = var.sagemaker.gh_repo_url
  }
}

resource "aws_sagemaker_notebook_instance" "notebook_instance" {
  name                    = var.sagemaker.notebook_instance_name
  role_arn                = aws_iam_role.aws_sagemaker_role.arn
  instance_type           = var.sagemaker.notebook_instance_type
  default_code_repository = aws_sagemaker_code_repository.gh_repo.code_repository_name

  tags = {
    Name = var.sagemaker.notebook_instance_name
  }
}

output "sagemaker_domain_id" {
  value = aws_sagemaker_domain.sagemaker_domain.id
}
