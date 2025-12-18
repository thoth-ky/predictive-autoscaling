resource "aws_vpc" "sagemaker_vpc" {
  cidr_block = "10.0.0.0/16"
    tags = {
        Name = "sagemaker-vpc"
    }
}

resource "aws_subnet" "sagemaker_subnet" {
  vpc_id            = aws_vpc.sagemaker_vpc.id
  cidr_block        = "10.0.1.0/24"
  availability_zone = "us-east-1a"

  tags = {
    Name = "sagemaker-subnet"
  }
}

# Internet Gateway for the VPC
resource "aws_internet_gateway" "sagemaker_igw" {
  vpc_id = aws_vpc.sagemaker_vpc.id

  tags = {
    Name = "sagemaker-igw"
  }
}

# Route table for the subnet
resource "aws_route_table" "sagemaker_rt" {
  vpc_id = aws_vpc.sagemaker_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.sagemaker_igw.id
  }

  tags = {
    Name = "sagemaker-rt"
  }
}

# Associate the route table with the subnet
resource "aws_route_table_association" "sagemaker_rta" {
  subnet_id      = aws_subnet.sagemaker_subnet.id
  route_table_id = aws_route_table.sagemaker_rt.id
}