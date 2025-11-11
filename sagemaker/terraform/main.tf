terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "6.3.0"
    }
    
  }

  backend "local" {}
}

variable "aws" {
  description = "AWS configuration"
  type = object({
    region  = string
    profile = string
  })
}

provider "aws" {
  region  = var.aws.region
  profile = var.aws.profile
}