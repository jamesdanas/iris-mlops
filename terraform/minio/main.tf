# WHY a variable:
#   Bucket names should be configurable, not hardcoded. This lets
#   you use different names in different environments (dev vs. prod).
variable "bucket_name" {
  description = "Name of the demo bucket"
  type        = string
  default     = "iris-mlops-terraform-demo"
}

# WHY a resource block:
#   Declares a MinIO bucket. Terraform creates it if it does not exist,
#   and tracks it in state. If the bucket drifts (deleted manually),
#   `terraform plan` detects it and offers to recreate.
resource "minio_s3_bucket" "demo" {
  bucket = var.bucket_name

  # WHY force_destroy:
  #   Allows `terraform destroy` to delete a non-empty bucket.
  #   Useful for learning; set to false in production to prevent
  #   accidental data loss.
  force_destroy = true
}

# WHY outputs:
#   Outputs surface useful values after `terraform apply`. Other
#   Terraform modules or shell scripts can consume them.
output "bucket_name" {
  value = minio_s3_bucket.demo.bucket
}
