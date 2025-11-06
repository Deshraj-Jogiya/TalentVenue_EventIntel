# ============================================================================
# ENTERPRISE DATA CONSOLIDATION & INTELLIGENCE FRAMEWORK
# Infrastructure as Code (IaC) - Terraform Configuration
# Target Cloud: Microsoft Azure & Snowflake Data Platform
# ============================================================================

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    snowflake = {
      source  = "Snowflake-Labs/snowflake"
      version = "~> 0.80"
    }
  }
}

# Configure Azure Provider
provider "azurerm" {
  features {}
}

# Configure Snowflake Provider
# Credentials can be loaded via environment variables: SNOWFLAKE_USER, SNOWFLAKE_PASSWORD, SNOWFLAKE_ACCOUNT
provider "snowflake" {
  role = "ACCOUNTADMIN"
}

# ----------------------------------------------------------------------------
# 1. Azure Infrastructure: Storage Container for Data Staging (ADLS Gen2)
# ----------------------------------------------------------------------------

resource "azurerm_resource_group" "dw_rg" {
  name     = "rg-enterprise-data-dw"
  location = "East US 2"
}

resource "azurerm_storage_account" "dw_storage" {
  name                     = "stenterpriseadlsstage"
  resource_group_name      = azurerm_resource_group.dw_rg.name
  location                 = azurerm_resource_group.dw_rg.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  account_kind             = "StorageV2"
  is_hns_enabled           = true # Enable Hierarchical Namespace (ADLS Gen2)

  tags = {
    Environment = "Bootcamp_Submission"
    Project     = "Data_Consolidation_Framework"
  }
}

resource "azurerm_storage_container" "stage_container" {
  name                  = "staged"
  storage_account_name  = azurerm_storage_account.dw_storage.name
  container_access_type = "private"
}

# ----------------------------------------------------------------------------
# 2. Snowflake Infrastructure: Data Warehouse Databases & Schemas
# ----------------------------------------------------------------------------

resource "snowflake_database" "enterprise_dw" {
  name    = "ENTERPRISE_DW"
  comment = "Enterprise Data Warehouse for Event and Talent Management"
}

resource "snowflake_schema" "raw_schema" {
  database = snowflake_database.enterprise_dw.name
  name     = "RAW"
  comment  = "Raw landing zone for staged files"
}

resource "snowflake_schema" "analytics_schema" {
  database = snowflake_database.enterprise_dw.name
  name     = "ANALYTICS"
  comment  = "Transformed star-schema data models (Facts & Dimensions)"
}

# ----------------------------------------------------------------------------
# 3. Security: Snowflake Storage Integration to Azure
# ----------------------------------------------------------------------------

resource "snowflake_storage_integration" "azure_adls_integration" {
  name    = "AZURE_ADLS_INTEGRATION"
  comment = "Secure storage integration connecting Snowflake to Azure ADLS Gen2 staged container"
  type    = "EXTERNAL_STAGE"

  enabled = true

  storage_provider      = "AZURE"
  azure_tenant_id       = "00000000-0000-0000-0000-000000000000" # Replace with Azure Active Directory Tenant ID
  storage_allowed_locations = [
    "azure://${azurerm_storage_account.dw_storage.name}.blob.core.windows.net/${azurerm_storage_container.stage_container.name}/"
  ]
}

# ----------------------------------------------------------------------------
# 4. Warehouse & Computing Resources
# ----------------------------------------------------------------------------

resource "snowflake_warehouse" "query_wh" {
  name           = "ANALYTICS_WH"
  warehouse_size = "X-SMALL"
  auto_suspend   = 60 # Suspend warehouse after 1 minute of inactivity (cost control)
  auto_resume    = true
  initially_suspended = true
}
