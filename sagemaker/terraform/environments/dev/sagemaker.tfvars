aws = {
  region  = "us-east-1"  # Change to your preferred region
  profile = "default"    # Change to your AWS profile name
}

sagemaker = {
  domain_name = "predictive-autoscaling-dev-domain"
  notebook_instance_name = "predictive-autoscaling-dev-notebook"
  notebook_instance_type = "ml.t2.medium"
  space_name  = "shared-space"
  user_profile_name = "default"
  gh_repo_url = "https://github.com/thoth-ky/predictive-autoscaling.git"
}