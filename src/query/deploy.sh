#!/bin/bash

FUNCTION_APP_NAME="chem-reasoner-query"
RESOURCE_GROUP="aqe-ldrd"

# Check if already logged in to Azure
if ! az account show &> /dev/null
then
    echo "Logging in to Azure..."
    az login
else
    echo "Already logged in to Azure."
fi

# Check if Azure Functions Core Tools are installed
if ! command -v func &> /dev/null
then
    echo "Azure Functions Core Tools not found, installing..."
    npm install -g azure-functions-core-tools@3 --unsafe-perm true
else
    echo "Azure Functions Core Tools already installed."
fi

# Bundle the Azure Function
echo "Bundling the Azure Function..."
zip function.zip *.py *.ini requirements.txt host.json

# Deploy the Azure Function
echo "Deploying the Azure Function..."
az functionapp deployment source config-zip \
    -g $RESOURCE_GROUP \
    -n $FUNCTION_APP_NAME \
    --src ./function.zip

echo "Deployment completed."