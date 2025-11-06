terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "6.3.0"
    }
    
  }
  backend "s3" {
    bucket = "mlops-tf-state-bucket"
  }
}


provider "aws" {
  region  = var.aws.region
  profile = var.aws.profile
}
