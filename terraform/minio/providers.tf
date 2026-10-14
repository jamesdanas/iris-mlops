# WHY a required_providers block:
#   Terraform downloads providers from the registry on `terraform init`.
#   Pinning versions ensures everyone gets the same provider behavior.
terraform {
  required_providers {
    minio = {
      source  = "aminueza/minio"
      version = "~> 3.0"
    }
  }
}

# WHY provider-level config:
#   The MinIO provider needs to know where MinIO is and how to authenticate.
#   These are hardcoded here for a local demo. In production, read them
#   from environment variables or a secrets manager.
provider "minio" {
  minio_server   = "localhost:9000"
  minio_user     = "minioadmin"
  minio_password = "minioadmin"
  minio_ssl      = false
}
